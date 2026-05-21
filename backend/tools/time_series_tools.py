from __future__ import annotations

import json

import numpy as np
import pandas as pd


class TimeSeriesTools:

    def __init__(self, df_source, memory=None):
        self._source = df_source
        self.memory = memory

    @property
    def df(self) -> pd.DataFrame:
        if hasattr(self._source, "df"):
            return self._source.df
        return self._source

    # ============================================================
    # TIME SERIES ANALYSIS
    # ============================================================

    def time_series_analysis(
        self,
        date_col: str,
        value_col: str,
    ) -> str:
        df = self.df.copy()

        if date_col not in df.columns:
            return f"Date column not found: {date_col}"
        if value_col not in df.columns:
            return f"Value column not found: {value_col}"

        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            return f"Could not parse '{date_col}' as datetime."

        ts = df[[date_col, value_col]].dropna().sort_values(date_col)
        if len(ts) < 3:
            return "Not enough rows for time series analysis (minimum 3)."

        series = ts.set_index(date_col)[value_col].sort_index()
        monthly = series.resample("ME").mean()
        growth = monthly.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        first_val = float(series.iloc[0])
        last_val = float(series.iloc[-1])
        trend = "increasing" if last_val > first_val else "decreasing"
        total_change_pct = round(float((last_val - first_val) / first_val * 100), 2) if first_val != 0 else None
        avg_growth = round(float(growth.mean() * 100), 2) if len(growth) else None
        volatility = round(float(growth.std() * 100), 2) if len(growth) > 1 else None

        result = {
            "date_column": date_col,
            "value_column": value_col,
            "rows_used": int(len(series)),
            "date_range": {
                "start": str(series.index.min()),
                "end": str(series.index.max()),
            },
            "trend": trend,
            "first_value": round(first_val, 4),
            "last_value": round(last_val, 4),
            "total_change_pct": total_change_pct,
            "avg_monthly_growth_pct": avg_growth,
            "volatility_pct": volatility,
            "peak": {
                "date": str(series.idxmax()),
                "value": round(float(series.max()), 4),
            },
            "lowest": {
                "date": str(series.idxmin()),
                "value": round(float(series.min()), 4),
            },
            "monthly_summary": {str(k): round(float(v), 4) for k, v in monthly.dropna().items()},
        }

        if self.memory:
            self.memory.add_finding(f"'{value_col}' shows a {trend} trend over time")
            if total_change_pct is not None:
                self.memory.add_finding(f"'{value_col}' changed by {total_change_pct}% over the observed period")

        return json.dumps(result, indent=2, default=str)

    # ============================================================
    # NAIVE FORECAST
    # ============================================================

    def naive_forecast(
        self,
        date_col: str,
        value_col: str,
        periods: int = 6,
    ) -> str:
        df = self.df.copy()

        if date_col not in df.columns:
            return f"Date column not found: {date_col}"
        if value_col not in df.columns:
            return f"Value column not found: {value_col}"

        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            return f"Could not parse '{date_col}' as datetime."

        ts = df[[date_col, value_col]].dropna().sort_values(date_col)
        if len(ts) < 5:
            return "Not enough rows for forecasting (minimum 5)."

        series = ts.set_index(date_col)[value_col].resample("ME").mean().dropna()
        if len(series) < 3:
            return "Not enough monthly data for forecasting."

        avg_change = float(series.diff().dropna().mean())
        last_date = series.index[-1]
        last_value = float(series.iloc[-1])

        forecasts = []
        current = last_value
        for i in range(periods):
            current += avg_change
            next_date = last_date + pd.DateOffset(months=i + 1)
            forecasts.append({"date": str(next_date.date()), "forecast": round(current, 4)})

        if self.memory:
            self.memory.add_finding(f"Generated {periods}-period naive forecast for '{value_col}'")

        return json.dumps({
            "value_column": value_col,
            "forecast_periods": periods,
            "latest_actual": round(last_value, 4),
            "average_period_change": round(avg_change, 4),
            "forecast": forecasts,
        }, indent=2)
