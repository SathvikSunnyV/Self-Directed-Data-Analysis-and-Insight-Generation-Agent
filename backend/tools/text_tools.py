from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd


class TextTools:

    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
        "was", "were", "are", "you", "your", "about", "there", "their", "would",
        "could", "should", "into", "than", "then", "them", "they", "what",
        "when", "where", "which", "while", "will", "shall", "can", "not",
        "but", "also", "very", "much", "more", "most", "some", "many",
        "any", "all", "out", "who", "why", "how", "its",
    }

    def __init__(self, df_source, memory=None):
        self._source = df_source
        self.memory = memory

    @property
    def df(self) -> pd.DataFrame:
        if hasattr(self._source, "df"):
            return self._source.df
        return self._source

    # ============================================================
    # TEXT FREQUENCY
    # ============================================================

    def text_frequency(
        self,
        col: str,
        top_n: int = 20,
    ) -> str:
        if col not in self.df.columns:
            return f"Column not found: {col}"

        text_series = self.df[col].dropna().astype(str)
        full_text = " ".join(text_series.tolist()).lower()
        words = re.findall(r"[a-zA-Z0-9']+", full_text)

        filtered = [w for w in words if len(w) >= 3 and w not in self.STOPWORDS]
        counts = Counter(filtered)
        top_words = counts.most_common(top_n)

        if self.memory and top_words:
            self.memory.add_finding(f"Most frequent word in '{col}': '{top_words[0][0]}'")

        return json.dumps({
            "column": col,
            "total_words": len(filtered),
            "unique_words": len(counts),
            "top_words": [{"word": w, "count": c} for w, c in top_words],
        }, indent=2)

    # ============================================================
    # TEXT LENGTH ANALYSIS
    # ============================================================

    def text_length_analysis(
        self,
        col: str,
    ) -> str:
        if col not in self.df.columns:
            return f"Column not found: {col}"

        s = self.df[col].dropna().astype(str)
        lengths = s.str.len()
        word_counts = s.str.split().str.len()

        if self.memory:
            self.memory.add_finding(
                f"Average text length in '{col}': {round(float(lengths.mean()), 2)} characters"
            )

        return json.dumps({
            "column": col,
            "rows_analyzed": int(len(s)),
            "character_stats": {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": round(float(lengths.mean()), 2),
                "median": round(float(lengths.median()), 2),
            },
            "word_stats": {
                "min": int(word_counts.min()),
                "max": int(word_counts.max()),
                "mean": round(float(word_counts.mean()), 2),
                "median": round(float(word_counts.median()), 2),
            },
        }, indent=2)

    # ============================================================
    # SIMPLE SENTIMENT
    # ============================================================

    def simple_sentiment(
        self,
        col: str,
    ) -> str:
        if col not in self.df.columns:
            return f"Column not found: {col}"

        POSITIVE = {
            "good", "great", "excellent", "amazing", "awesome", "best", "love", "happy",
            "positive", "satisfied", "success", "fast", "easy", "helpful", "perfect",
        }
        NEGATIVE = {
            "bad", "poor", "terrible", "awful", "worst", "hate", "negative", "slow",
            "hard", "problem", "issue", "bug", "error", "difficult", "disappointed",
        }

        texts = self.df[col].dropna().astype(str).str.lower()
        positive, negative, neutral = 0, 0, 0

        for text in texts:
            words = set(re.findall(r"[a-zA-Z0-9']+", text))
            pos = len(words & POSITIVE)
            neg = len(words & NEGATIVE)
            if pos > neg:
                positive += 1
            elif neg > pos:
                negative += 1
            else:
                neutral += 1

        total = max(len(texts), 1)
        dominant = max([("positive", positive), ("negative", negative), ("neutral", neutral)], key=lambda x: x[1])[0]

        if self.memory:
            self.memory.add_finding(f"Dominant sentiment in '{col}': {dominant}")

        return json.dumps({
            "column": col,
            "rows_analyzed": int(len(texts)),
            "positive": {"count": positive, "pct": round(positive / total * 100, 2)},
            "negative": {"count": negative, "pct": round(negative / total * 100, 2)},
            "neutral": {"count": neutral, "pct": round(neutral / total * 100, 2)},
            "dominant_sentiment": dominant,
        }, indent=2)
