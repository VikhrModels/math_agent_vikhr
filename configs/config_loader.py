"""
Centralized configuration loader for Math Agent Vikhr project.
Loads configuration from YAML file and provides the same interface as before.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Any, Dict

def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in config values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        # Extract variable name and default value
        var_part = value[2:-1]  # Remove ${ and }
        if ":-" in var_part:
            var_name, default_value = var_part.split(":-", 1)
        else:
            var_name, default_value = var_part, None
        
        # Get from environment or use default
        env_value = os.environ.get(var_name)
        return env_value if env_value is not None else default_value
    
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    
    else:
        return value

def _resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Resolve relative paths to absolute paths."""
    resolved_config = {}
    
    for key, value in config.items():
        if key == "paths":
            # Handle paths section specially
            paths = {}
            for path_key, path_value in value.items():
                if path_key == "base_dir":
                    paths[path_key] = base_dir
                else:
                    # Convert relative paths to absolute
                    if isinstance(path_value, str):
                        paths[path_key] = base_dir / path_value
                    else:
                        paths[path_key] = path_value
            resolved_config[key] = paths
        elif key == "lean":
            # Handle lean section
            lean = {}
            for lean_key, lean_value in value.items():
                if lean_key == "source_file":
                    # Source file is relative to minif2f_dir
                    lean[lean_key] = base_dir / "miniF2F-lean4" / lean_value
                elif lean_key == "output_file":
                    # Output file is relative to base_dir
                    lean[lean_key] = base_dir / lean_value
                else:
                    lean[lean_key] = lean_value
            resolved_config[key] = lean
        elif key == "logging":
            # Handle logging section
            logging = {}
            for log_key, log_value in value.items():
                if log_key == "file":
                    # Log file is relative to log_dir
                    logging[log_key] = base_dir / "log" / log_value
                else:
                    logging[log_key] = log_value
            resolved_config[key] = logging
        else:
            resolved_config[key] = value
    
    return resolved_config

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get base directory (parent of configs folder)
    base_dir = config_path.parent.parent
    
    # Resolve environment variables
    config = _resolve_env_vars(config)
    
    # Resolve paths
    config = _resolve_paths(config, base_dir)
    
    return config

# Load configuration when module is imported
_config = load_config()

# --- Export commonly used configurations with the same names as before ---
# Paths
BASE_DIR = _config['paths']['base_dir']
MINIF2F_DIR = _config['paths']['minif2f_dir']
LOG_DIR = _config['paths']['log_dir']
TMP_DIR = _config['paths']['tmp_dir']

# API Configuration
OPENROUTER_API_KEY = _config['api']['openrouter']['api_key']
OPENROUTER_API_BASE = _config['api']['openrouter']['api_base']
OPENAI_API_KEY = _config['api']['openai']['api_key']
OPENAI_API_BASE = _config['api']['openai']['api_base']
DEFAULT_PROVIDER = _config['api']['default_provider']
AVAILABLE_PROVIDERS = _config['api']['available_providers']

# Models
DEFAULT_MODEL = _config['models']['default']

# Lean Configuration
LEAN_SOURCE_FILE = _config['lean']['source_file']
LEAN_OUTPUT_FILE = _config['lean']['output_file']
LEAN_TIMEOUT = _config['lean']['timeout']

# LLM/API timeouts
LLM_REQUEST_TIMEOUT = _config['llm']['request_timeout']

# Agent Configuration
DEFAULT_SUBSET_SIZE = _config['agent']['default_subset_size']
DEFAULT_LOG_LEVEL = _config['agent']['default_log_level']
DEFAULT_JSON_FILE = _config['agent']['default_json_file']
DEFAULT_MAX_STEPS = _config['agent']['max_steps']
DEFAULT_PLANNING_INTERVAL = _config['agent']['planning_interval']
DEFAULT_CONCURRENCY = _config['agent']['concurrency']

# Agent Budgets Configuration
AGENT_BUDGETS = _config['agent']['budgets']

# Logging Configuration
LOG_FILE = _config['logging']['file']
LOG_FORMAT = _config['logging']['format']

# --- Validation functions ---
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
    "AGENT_BUDGETS",
    "LOG_FILE",
    "LOG_FORMAT",
    "validate_config",
    "validate_provider_credentials",
    "get_model_for_environment",
    "get_subset_size_for_environment",
] 