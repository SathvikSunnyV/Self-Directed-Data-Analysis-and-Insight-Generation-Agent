from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

import uvicorn

from agents.agent import AutonomousAgent
from agents.schema import SchemaInspector
from agents.llm import HuggingFaceLLM, DEFAULT_MODEL, FALLBACK_MODEL


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Autonomous Data Analytics Agent",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs",
)

# =============================================================================
# MEMORY DATASET STORE
# =============================================================================

DATASETS: dict[str, pd.DataFrame] = {}

schema_inspector = SchemaInspector()

# =============================================================================
# HELPERS
# =============================================================================


def json_safe(obj: Any):

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)

    if obj is pd.NaT:
        return None

    return obj


def sse(event: dict) -> str:
    return f"data: {json.dumps(json_safe(event))}\n\n"


def load_dataframe(path: Path, filename: str) -> pd.DataFrame:

    ext = filename.lower().split(".")[-1]

    if ext == "csv":
        df = pd.read_csv(path)

    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(path)

    elif ext == "json":
        df = pd.read_json(path)

    elif ext == "parquet":
        df = pd.read_parquet(path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # deduplicate columns
    if df.columns.duplicated().any():

        seen = {}
        new_cols = []

        for col in df.columns.astype(str):

            seen[col] = seen.get(col, 0) + 1

            if seen[col] == 1:
                new_cols.append(col)
            else:
                new_cols.append(f"{col}_{seen[col]}")

        df.columns = new_cols

    return df


def quick_profile(df: pd.DataFrame, filename: str) -> dict:

    profile = {
        "name": filename,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": [],
    }

    for col in df.columns:

        s = df[col]
        nn = s.dropna()

        info = {
            "name": col,
            "dtype": str(s.dtype),
            "unique": int(s.nunique()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
        }

        if pd.api.types.is_numeric_dtype(s):

            info["type"] = "numeric"

            if len(nn):

                info.update({
                    "min": round(float(nn.min()), 4),
                    "max": round(float(nn.max()), 4),
                    "mean": round(float(nn.mean()), 4),
                    "std": round(float(nn.std()), 4),
                })

        elif pd.api.types.is_datetime64_any_dtype(s):

            info["type"] = "datetime"

        else:

            info["type"] = "categorical"

            info["top_values"] = {
                str(k): int(v)
                for k, v in s.value_counts(dropna=False).head(5).items()
            }

        profile["columns"].append(info)

    return profile


# =============================================================================
# ROUTES
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():

    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    return HTMLResponse("<h1>Frontend not found</h1>")


@app.get("/health")
async def health():
    llm = HuggingFaceLLM(model=DEFAULT_MODEL)
    available = llm.is_available()
    active_model = llm.model if available else None
    return json_safe({
        "status": "ok",
        "llm_backend": "huggingface",
        "llm_available": available,
        "active_model": active_model,
        "supported_models": llm.list_models(),
        "datasets_loaded": list(DATASETS.keys()),
    })


@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...)
):

    results = []

    for file in files:

        try:

            unique_name = f"{uuid.uuid4().hex}_{file.filename}"

            save_path = UPLOAD_DIR / unique_name

            save_path.write_bytes(await file.read())

            df = load_dataframe(save_path, file.filename)

            DATASETS[file.filename] = df

            schema = schema_inspector.inspect(
                df,
                filename=file.filename,
            )

            profile = quick_profile(
                df,
                filename=file.filename,
            )

            results.append({
                "status": "ok",
                "filename": file.filename,
                "shape": [len(df), len(df.columns)],
                "profile": profile,
                "schema": {
                    "domain": schema["domain"],
                    "numeric_cols": schema["numeric_cols"],
                    "categorical_cols": schema["categorical_cols"],
                    "datetime_cols": schema["datetime_cols"],
                    "text_cols": schema["text_cols"],
                    "potential_targets": schema["potential_targets"],
                },
            })

        except Exception as e:

            results.append({
                "status": "error",
                "filename": file.filename,
                "error": str(e),
            })

    return JSONResponse(
        content=json_safe({
            "uploads": results,
        })
    )


@app.get("/datasets")
async def list_datasets():

    out = []

    for name, df in DATASETS.items():

        out.append({
            "name": name,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": list(df.columns),
        })

    return json_safe({
        "datasets": out,
    })


@app.post("/analyze/stream")
async def analyze_stream(
    datasets: str = Form(...),
    question: str = Form(default=""),
    model: str = Form(default=DEFAULT_MODEL),
    hf_token: str = Form(default=""),
):

    dataset_names = [
        d.strip()
        for d in datasets.split(",")
        if d.strip()
    ]

    missing = [
        d
        for d in dataset_names
        if d not in DATASETS
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Datasets not loaded: {missing}",
        )

    async def event_stream():

        try:

            yield sse({
                "type": "start",
                "message": "Initializing autonomous agent...",
                "datasets": dataset_names,
            })

            await asyncio.sleep(0.05)

            # ============================================================
            # CURRENTLY SINGLE DATASET
            # ============================================================

            dataset_name = dataset_names[0]

            df = DATASETS[dataset_name]

            loop = asyncio.get_event_loop()

            result_box = {}
            error_box = {}

            progress_queue: asyncio.Queue = asyncio.Queue()

            # ============================================================
            # PROGRESS CALLBACK
            # ============================================================

            def progress_cb(
                agent_name: str,
                message: str,
                step: int,
                total: int,
            ):

                try:

                    loop.call_soon_threadsafe(
                        progress_queue.put_nowait,
                        {
                            "type": "progress",
                            "agent": agent_name,
                            "message": message,
                            "step": step,
                            "total": total,
                        },
                    )

                except Exception:
                    pass

            # ============================================================
            # RUN AGENT THREAD
            # ============================================================

            def run_agent():

                try:

                    agent = AutonomousAgent(
                        model=model,
                        hf_token=hf_token or None,
                    )

                    result_box["result"] = agent.analyze(
                        df=df,
                        filename=dataset_name,
                        user_question=question,
                        progress_cb=progress_cb,
                    )

                except Exception as e:

                    error_box["error"] = str(e)

                    error_box["traceback"] = traceback.format_exc()

            task = loop.run_in_executor(
                None,
                run_agent,
            )

            # ============================================================
            # STREAM PROGRESS
            # ============================================================

            while not task.done():

                while True:

                    try:

                        event = progress_queue.get_nowait()

                        yield sse(event)

                    except asyncio.QueueEmpty:
                        break

                await asyncio.sleep(0.1)

            await task

            while not progress_queue.empty():

                try:

                    event = progress_queue.get_nowait()

                    yield sse(event)

                except Exception:
                    break

            # ============================================================
            # HANDLE ERROR
            # ============================================================

            if error_box:

                yield sse({
                    "type": "error",
                    "message": error_box["error"],
                    "traceback": error_box["traceback"],
                })

                return

            result = result_box["result"]

            # ============================================================
            # SEND RESULTS
            # ============================================================

            yield sse({
                "type": "complete",
                "message": "Analysis complete",
                "result": result,
            })

        except Exception as e:

            yield sse({
                "type": "fatal_error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.get("/download-report")
async def download_report(path: str):

    file_path = Path(path)

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Report not found",
        )

    return FileResponse(
        path=file_path,
        filename=file_path.name,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )