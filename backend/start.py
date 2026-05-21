from __future__ import annotations

import os
import sys
import socket
from pathlib import Path

import uvicorn
import httpx


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

FRONTEND_DIR = BASE_DIR.parent / "frontend"
FRONTEND_FILE = FRONTEND_DIR / "index.html"


# ============================================================
# COLORS
# ============================================================

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


# ============================================================
# HELPERS
# ============================================================

def log_info(msg: str):
    print(f"{BLUE}[INFO]{RESET} {msg}")


def log_success(msg: str):
    print(f"{GREEN}[SUCCESS]{RESET} {msg}")


def log_warn(msg: str):
    print(f"{YELLOW}[WARNING]{RESET} {msg}")


def log_error(msg: str):
    print(f"{RED}[ERROR]{RESET} {msg}")


# ============================================================
# CHECK PORT
# ============================================================

def is_port_in_use(port: int) -> bool:

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        return s.connect_ex(("127.0.0.1", port)) == 0


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    log_info("Checking Ollama connection...")

    try:

        response = httpx.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )

        if response.status_code != 200:

            log_warn("Ollama API responded with non-200 status")
            return False

        data = response.json()

        models = data.get("models", [])

        log_success(
            f"Ollama connected ({len(models)} models available)"
        )

        for model in models[:10]:

            name = model.get("name", "unknown")

            print(f"   - {name}")

        return True

    except Exception as e:

        log_warn(f"Ollama not running: {e}")

        print()
        print(f"{YELLOW}Run this in another terminal:{RESET}")
        print("ollama serve")
        print()

        return False


# ============================================================
# CHECK FILES
# ============================================================

def check_project_structure():

    log_info("Checking project structure...")

    required_dirs = [
        UPLOAD_DIR,
        OUTPUTS_DIR,
        FRONTEND_DIR,
    ]

    for d in required_dirs:

        if not d.exists():

            log_warn(f"Creating missing directory: {d}")

            d.mkdir(parents=True, exist_ok=True)

    if FRONTEND_FILE.exists():

        log_success("Frontend found")

    else:

        log_error("frontend/index.html not found")

    required_files = [
        BASE_DIR / "main.py",
        BASE_DIR / "agents" / "agent.py",
        BASE_DIR / "agents" / "__init__.py",
    ]

    missing = []

    for f in required_files:

        if not f.exists():

            missing.append(str(f))

    if missing:

        log_error("Missing required files:")

        for m in missing:

            print(f"   - {m}")

        return False

    log_success("Core files verified")

    return True


# ============================================================
# START SERVER
# ============================================================

def start_server():

    port = 8000

    if is_port_in_use(port):

        log_warn(f"Port {port} already in use")

        print()
        print("Close the existing process or use another port.")
        print()

        return

    log_success("Starting Autonomous Analytics Agent")

    print()
    print(f"{GREEN}Backend:{RESET}  http://localhost:8000")
    print(f"{GREEN}Frontend:{RESET} http://localhost:8000")
    print(f"{GREEN}Docs:{RESET}     http://localhost:8000/docs")
    print()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" AUTONOMOUS DATA ANALYTICS AGENT ")
    print("=" * 70)
    print()

    ok = check_project_structure()

    if not ok:

        print()
        log_error("Project structure invalid")
        sys.exit(1)

    print()

    check_ollama()

    print()

    start_server()


if __name__ == "__main__":

    main()