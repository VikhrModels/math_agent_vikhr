"""
Tools module for Math Agent Vikhr.

This module contains custom tools and utilities for the mathematical theorem proving agent.
Currently contains imports and basic setup for future tool implementations.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import configuration
from config import (
    MINIF2F_DIR,
    TMP_DIR,
    LEAN_TIMEOUT,
    LOG_FILE,
    LOG_FORMAT
)

# Import smolagents for tool definitions
try:
    from smolagents import tool, Tool
    from smolagents.tools import BaseTool
except ImportError:
    print("Warning: smolagents not available. Tools will not be functional.")
    tool = None
    Tool = None
    BaseTool = None

# Import logging
import logging

# Set up logging for tools
logger = logging.getLogger(__name__)

# Common utility functions that might be used by tools
def create_temp_lean_file(content: str, theorem_name: str) -> Path:
    """Create a temporary Lean file for theorem verification."""
    temp_file = TMP_DIR / f"{theorem_name}_temp.lean"
    temp_file.write_text(content, encoding='utf-8')
    return temp_file

def run_lean_verification(lean_file: Path) -> Dict[str, Any]:
    """Run Lean verification on a file and return results."""
    try:
        result = subprocess.run(
            ["lean", "--make", str(lean_file)],
            cwd=MINIF2F_DIR,
            capture_output=True,
            text=True,
            timeout=LEAN_TIMEOUT
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Lean verification timed out",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }

# Export common utilities
__all__ = [
    "create_temp_lean_file",
    "run_lean_verification",
    "logger",
    "tool",
    "Tool", 
    "BaseTool",
] 