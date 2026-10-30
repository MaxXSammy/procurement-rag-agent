"""Zentrale Einstellungen. Werte lassen sich über Umgebungsvariablen oder eine .env-Datei setzen."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv ist optional
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "dokumente"
GOLD_DIR = DATA_DIR / "gold"
BUSINESS_FILE = DATA_DIR / "business.json"
CACHE_DIR = DATA_DIR / "cache"
HISTORY_FILE = DATA_DIR / "eval_verlauf.jsonl"

# Modus: "auto" nutzt Ollama, wenn erreichbar, sonst den Offline-Modus.
MODUS = os.getenv("AGENT_MODUS", "auto")  # auto | ollama | offline
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Retrieval
CHUNK_SIZE = 600  # Zeichen
CHUNK_OVERLAP_LINES = 1
TOP_K = 4

# Agent
MAX_AGENT_SCHRITTE = 4
