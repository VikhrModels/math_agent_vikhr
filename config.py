"""
Centralized configuration file for Math Agent Vikhr project.
Contains all configurable parameters that may change between runs, environments,
or require convenient access.
"""

import os
from pathlib import Path
from typing import Optional

# --- Project Paths ---
BASE_DIR = Path(__file__).parent
MINIF2F_DIR = BASE_DIR / "miniF2F-lean4"
LOG_DIR = BASE_DIR / "log"
TMP_DIR = BASE_DIR / "tmp"

# --- API Configuration ---
# OpenRouter (proxy for many providers)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# OpenAI (direct provider)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.openai.com/v1"

# Default provider selection
DEFAULT_PROVIDER = os.environ.get("MATH_AGENT_PROVIDER", "openrouter")  # one of: ["openrouter", "openai"]
AVAILABLE_PROVIDERS = ["openrouter", "openai"]

# --- Model Configuration ---
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
AVAILABLE_MODELS = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "openai/gpt-4.1",
]

# --- Lean Configuration ---
# In Lean 4 project, validation statements reside in `MiniF2F/Valid.lean`
LEAN_SOURCE_FILE = MINIF2F_DIR / "MiniF2F" / "Valid.lean"
LEAN_OUTPUT_FILE = BASE_DIR / "valid.json"
LEAN_TIMEOUT = 900  # seconds for Lean compilation (increased from 30)

# --- LLM/API timeouts ---
# Default HTTP request timeout for LLM API calls (seconds)
LLM_REQUEST_TIMEOUT = int(os.environ.get("LLM_REQUEST_TIMEOUT", "120"))

# --- Agent Configuration ---
DEFAULT_SUBSET_SIZE = 10
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_JSON_FILE = "valid.json"

# Agent step limit
DEFAULT_MAX_STEPS = 5  # Maximum number of agent steps per theorem

# Planning interval for agent (how often to run planning phase)
DEFAULT_PLANNING_INTERVAL = 2  # Run planning every N steps

# --- Concurrency ---
DEFAULT_CONCURRENCY = 4 # Number of theorems to process in parallel

# --- Logging Configuration ---
LOG_FILE = LOG_DIR / "llm_requests.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- Validation ---
def validate_config() -> None:
    """Validate static configuration that is provider-agnostic.

    API credentials are validated at runtime based on the selected provider.
    """
    if not LEAN_SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Lean source file not found at {LEAN_SOURCE_FILE}. "
            "Make sure miniF2F-lean4 project is present."
        )

def validate_provider_credentials(provider: str) -> None:
    """Validate API credentials for a given provider."""
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# --- Environment-specific overrides ---
def get_model_for_environment() -> str:
    """Get the appropriate model based on environment."""
    # You can add environment-specific logic here
    # For example, use a cheaper model in development
    return os.environ.get("MATH_AGENT_MODEL", DEFAULT_MODEL)

def get_subset_size_for_environment() -> int:
    """Get the appropriate subset size based on environment."""
    # Use smaller subset in development for faster iteration
    return int(os.environ.get("MATH_AGENT_SUBSET_SIZE", str(DEFAULT_SUBSET_SIZE)))

# --- Export commonly used configurations ---
__all__ = [
    "BASE_DIR",
    "MINIF2F_DIR", 
    "LOG_DIR",
    "TMP_DIR",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_API_BASE",
    "DEFAULT_PROVIDER",
    "AVAILABLE_PROVIDERS",
    "DEFAULT_MODEL",
    "AVAILABLE_MODELS",
    "LEAN_SOURCE_FILE",
    "LEAN_OUTPUT_FILE",
    "LEAN_TIMEOUT",
    "LLM_REQUEST_TIMEOUT",
    "DEFAULT_SUBSET_SIZE",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_JSON_FILE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_PLANNING_INTERVAL",
    "DEFAULT_CONCURRENCY",
    "LOG_FILE",
    "LOG_FORMAT",
    "validate_config",
    "validate_provider_credentials",
    "get_model_for_environment",
    "get_subset_size_for_environment",
] 