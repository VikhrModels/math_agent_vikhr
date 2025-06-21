"""
Tools module for Math Agent Vikhr.

This module contains custom tools and utilities for the mathematical theorem proving agent.
Currently contains imports and basic setup for future tool implementations.
"""

import os
import re
import subprocess
import tempfile
import time
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


class LeanCompilerTool(Tool):
    """
    A tool that compiles and verifies Lean code using the Lean 3.42.1 compiler.
    
    This tool takes complete Lean code (including theorem statements and proofs) and
    attempts to compile it using the Lean compiler within the miniF2F project context.
    It returns compilation success status and any output/errors from the compiler.
    """
    name = "lean_compiler"
    description = """
    Compiles and verifies Lean 3.42.1 code using the Lean compiler.
    Takes complete Lean code as input and returns compilation success status and output.
    The tool handles temporary file creation, compilation, and cleanup automatically.
    """
    inputs = {
        "lean_code": {
            "type": "string",
            "description": "Complete Lean code to compile, including theorem statements and proofs",
        }
    }
    output_type = "object"

    def forward(self, lean_code: str) -> Dict[str, Any]:
        """
        Compiles the provided Lean code and returns the result.
        
        Args:
            lean_code: Complete Lean code to compile
            
        Returns:
            Dictionary with 'success' (bool) and 'output' (str) fields
        """
        logger.info("Attempting Lean 3.42.1 compilation.")
        
        if not lean_code.strip():
            logger.warning("Empty Lean code provided.")
            return {"success": False, "output": "Empty Lean code provided."}

        # The miniF2F project structure is required for verification, as it contains
        # dependent libraries like mathlib. We create the temp file within its source directory.
        src_dir = MINIF2F_DIR / "lean" / "src"
        src_dir.mkdir(parents=True, exist_ok=True)  # Ensure src dir exists

        # Create a temporary file inside the miniF2F project's src directory
        # Use a fixed name to avoid cluttering the directory with failed attempts
        temp_file_name = f"temp_proof_{os.getpid()}_{time.time_ns()}.lean"
        temp_file_path = src_dir / temp_file_name
        olean_path = temp_file_path.with_suffix('.olean')

        # The `minif2f_import` file contains all necessary imports for the miniF2F theorems.
        # It must be included for the proof to be verifiable.
        lean_imports = "import minif2f_import\n\n"

        try:
            with temp_file_path.open("w", encoding='utf-8') as f:
                f.write(lean_imports)
                f.write(lean_code)

            # We run `lean --make` from the root of the miniF2F project.
            # This command compiles the file and its dependencies, which is the
            # standard way to build a Lean project file.
            process = subprocess.run(
                ["lean", "--make", f"lean/src/{temp_file_name}"],
                cwd=str(MINIF2F_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=LEAN_TIMEOUT
            )

            output = process.stdout + process.stderr

            # Log full Lean output for debugging
            logger.debug(f"Full Lean output for {temp_file_path.name}:\n{output}")

            if process.returncode != 0 or "error:" in output.lower():
                logger.warning(f"Compilation failed: Lean reported errors.")
                # Extract and log only error lines for clarity
                error_lines = [line for line in output.splitlines() if "error:" in line.lower()]
                if error_lines:
                    logger.warning("Lean errors:")
                    for error in error_lines:
                        logger.warning(f"  {error.strip()}")
                return {"success": False, "output": output}

            # For `lean --make`, a successful compilation often produces no output.
            # The presence of the .olean file and no errors is a good indicator of success.
            if not olean_path.exists():
                logger.warning(f"Compilation failed: .olean file was not produced, but no errors were reported.")
                return {"success": False, "output": "Compilation failed: .olean file was not produced, but no errors were reported."}

            logger.info(f"Compilation successful.")
            return {"success": True, "output": output}

        except FileNotFoundError:
            error_msg = "Lean executable 'lean' not found. Make sure Lean 3.42.1 is installed and in your PATH."
            logger.error(error_msg)
            return {"success": False, "output": error_msg}
        except subprocess.TimeoutExpired:
            error_msg = f"Lean compilation timed out after {LEAN_TIMEOUT} seconds."
            logger.error(error_msg)
            return {"success": False, "output": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred during Lean compilation: {e}"
            logger.error(error_msg)
            return {"success": False, "output": error_msg}
        finally:
            # Cleanup the temporary files
            if temp_file_path.exists():
                temp_file_path.unlink()
            if olean_path.exists():
                olean_path.unlink()


# Create tool instance
lean_compiler_tool = LeanCompilerTool()


# Export common utilities
__all__ = [
    "create_temp_lean_file",
    "run_lean_verification",
    "logger",
    "tool",
    "Tool", 
    "BaseTool",
    "LeanCompilerTool",
    "lean_compiler_tool",
] 