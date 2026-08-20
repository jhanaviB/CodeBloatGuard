import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

JUDGE_PROVIDER = os.getenv("CBG_PROVIDER", "gemini")

EMBED_MODEL = "gemini-embedding-001"

JUDGE_MODEL = "gemini-2.5-flash"
CONVENTIONS_MODEL = "gemini-2.5-flash"
TRIAGE_MODEL = "gemini-2.5-flash"

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

EMBED_BATCH = 50

DUP_DISTANCE = 0.30

# check-pr hands a function to the agent when its nearest neighbour lands in
# the gap between these two. Below DUP_DISTANCE the fast path already caught
# it; above this, nothing was ever close enough to be worth the calls.
ESCALATE_DISTANCE = 0.45

# Agent widening increment - how much to increase search radius when WIDEN is chosen
# Smaller values = more conservative expansion, more attempts needed
# Larger values = aggressive expansion, fewer attempts, might overshoot
WIDEN_STEP = 0.05

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}
