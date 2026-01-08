"""
Configuration for Retell AI Clone
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
CORE_DIR = BASE_DIR / 'core'
DATA_DIR = BASE_DIR / 'data'
SCRIPTS_DIR = DATA_DIR / 'scripts'
LOGS_DIR = DATA_DIR / 'logs'

# Create directories if they don't exist
for dir_path in [DATA_DIR, SCRIPTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
OLLAMA_TIMEOUT = 120
OLLAMA_TEMPERATURE = 0.1  # Low for more deterministic responses
OLLAMA_TOP_P = 0.9
OLLAMA_TOP_K = 40
OLLAMA_MAX_TOKENS = 300

# ============================================================================
# EMBEDDING MODEL (Semantic Search)
# ============================================================================
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_DEVICE = 'cpu'  # Change to 'cuda' if you have GPU

# ============================================================================
# INTENT DETECTION
# ============================================================================
INTENT_CONFIDENCE_THRESHOLD = 0.75
USE_FUZZY_MATCHING = True
FUZZY_THRESHOLD = 80  # Out of 100

# ============================================================================
# SEMANTIC MATCHING
# ============================================================================
SIMILARITY_THRESHOLD = 0.65  # Minimum similarity score
TOP_K_MATCHES = 3  # Number of script sections to retrieve

# ============================================================================
# VALIDATION
# ============================================================================
ENABLE_VALIDATION = True
MIN_RESPONSE_CONFIDENCE = 0.70
MAX_RESPONSE_LENGTH = 500
MIN_RESPONSE_LENGTH = 10

# ============================================================================
# CONVERSATION
# ============================================================================
MAX_CONVERSATION_HISTORY = 10  # Remember last N messages
CONTEXT_WINDOW = 5  # Use last N messages for context

# ============================================================================
# LOGGING
# ============================================================================
ENABLE_LOGGING = True
LOG_LEVEL = 'INFO'
LOG_FILE = LOGS_DIR / 'app.log'

# ============================================================================
# UI SETTINGS
# ============================================================================
PAGE_TITLE = "Retell AI Clone - Call Agent"
PAGE_ICON = "🤖"
LAYOUT = "wide"