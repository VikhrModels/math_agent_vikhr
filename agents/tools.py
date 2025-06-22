import sys
import os
import re
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import Tool
from config import MINIF2F_DIR, LEAN_TIMEOUT

# Set up logging for the tool
logger = logging.getLogger(__name__)

class VerifyLeanProof(Tool):
    """
    A tool for verifying Lean 3.42.1 mathematical proofs by compiling them within the miniF2F project environment.
    
    This tool takes a complete Lean theorem statement and attempts to compile it using the Lean 3.42.1 compiler.
    It creates a temporary file within the miniF2F project structure to ensure all necessary dependencies
    (like mathlib) are available during compilation.
    
    The tool is designed to work with theorems that have a `sorry` placeholder that needs to be replaced
    with an actual proof. It handles both `:= sorry` and `:= begin sorry end` formats.
    
    Returns a dictionary with:
    - success: boolean indicating whether compilation was successful
    - output: string containing the full compiler output (stdout + stderr)
    """
    
    name = "verify_lean_proof"
    description = """
    Verifies a Lean 3.42.1 mathematical proof by compiling it within the miniF2F project environment.
    
    This tool creates a temporary Lean file with the provided theorem and attempts to compile it
    using the Lean 3.42.1 compiler. It ensures all necessary dependencies are available by working
    within the miniF2F project structure.
    
    The theorem should be a complete Lean statement that may contain a `sorry` placeholder.
    The tool will attempt to compile the theorem and return whether it was successful.
    
    Args:
        theorem_statement: The complete Lean theorem statement to verify. This should be a full
                          theorem definition that may contain `:= sorry` or `:= begin sorry end`.
    
    Returns:
        A dictionary with:
        - success: boolean indicating whether the Lean compilation was successful
        - output: string containing the full compiler output (stdout + stderr)
    """
    
    def forward(self, theorem_statement: str) -> Dict[str, Any]:
        """
        Verifies a Lean proof by creating a temporary file and compiling it.
        
        Args:
            theorem_statement: Complete Lean theorem statement to verify
            
        Returns:
            Dictionary with 'success' boolean and 'output' string
        """
        logger.info("Starting Lean 3.42.1 proof verification")
        logger.info(f"Theorem statement length: {len(theorem_statement)} characters")
        
        # Validate input
        if not theorem_statement or not theorem_statement.strip():
            logger.warning("Empty theorem statement provided")
            return {
                "success": False,
                "output": "Error: Empty theorem statement provided"
            }
        
        # Check if the statement contains a sorry placeholder
        if "sorry" not in theorem_statement.lower():
            logger.warning("No 'sorry' placeholder found in theorem statement")
            logger.debug(f"Theorem statement: {theorem_statement[:200]}...")
            return {
                "success": False,
                "output": "Error: No 'sorry' placeholder found in theorem statement"
            }
        
        # Find the part of the statement before the proof begins
        # The statements typically end with `:= sorry` or `:= begin sorry end`
        statement_base_match = re.search(r'(.*):=\s*(begin\s*sorry\s*end|sorry)', theorem_statement, re.DOTALL)
        if not statement_base_match:
            logger.error("Could not find `:= sorry` or `:= begin sorry end` in the theorem statement")
            logger.debug(f"Full statement: {theorem_statement}")
            return {
                "success": False,
                "output": "Error: Could not find `:= sorry` or `:= begin sorry end` in the theorem statement"
            }
        
        statement_base = statement_base_match.group(1).strip()
        logger.info(f"Extracted statement base, length: {len(statement_base)} characters")
        
        # Create complete theorem with generated proof
        # Note: The theorem_statement should already contain the proof, not just the base
        full_lean_code = theorem_statement
        
        # Set up file paths
        src_dir = MINIF2F_DIR / "lean" / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file with unique name
        temp_file_name = f"temp_proof_{os.getpid()}_{time.time_ns()}.lean"
        temp_file_path = src_dir / temp_file_name
        olean_path = temp_file_path.with_suffix('.olean')
        
        logger.info(f"Creating temporary file: {temp_file_path}")
        
        # The minif2f_import file contains all necessary imports for the miniF2F theorems
        lean_imports = "import minif2f_import\n\n"
        
        try:
            # Write the complete Lean code to the temporary file
            with temp_file_path.open("w", encoding='utf-8') as f:
                f.write(lean_imports)
                f.write(full_lean_code)
            
            logger.info(f"Successfully wrote Lean code to {temp_file_path}")
            logger.debug(f"File size: {temp_file_path.stat().st_size} bytes")
            
            # Run lean --make from the root of the miniF2F project
            logger.info("Starting Lean compilation...")
            process = subprocess.run(
                ["lean", "--make", f"lean/src/{temp_file_name}"],
                cwd=str(MINIF2F_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=LEAN_TIMEOUT
            )
            
            # Combine stdout and stderr
            output = process.stdout + process.stderr
            logger.info(f"Lean compilation completed with return code: {process.returncode}")
            logger.debug(f"Full Lean output:\n{output}")
            
            # Check for compilation success
            if process.returncode != 0 or "error:" in output.lower():
                logger.warning("Lean compilation failed")
                # Extract and log error lines for clarity
                error_lines = [line for line in output.splitlines() if "error:" in line.lower()]
                if error_lines:
                    logger.warning("Lean errors found:")
                    for error in error_lines:
                        logger.warning(f"  {error.strip()}")
                
                return {
                    "success": False,
                    "output": output
                }
            
            # Check if .olean file was produced (indicator of successful compilation)
            if not olean_path.exists():
                logger.warning("Compilation appeared successful but .olean file was not produced")
                return {
                    "success": False,
                    "output": output + "\nWarning: .olean file was not produced despite no errors"
                }
            
            logger.info("Lean compilation successful!")
            return {
                "success": True,
                "output": output
            }
            
        except FileNotFoundError:
            error_msg = "Lean executable 'lean' not found. Make sure Lean 3.42.1 is installed and in your PATH."
            logger.error(error_msg)
            return {
                "success": False,
                "output": f"Error: {error_msg}"
            }
            
        except subprocess.TimeoutExpired:
            error_msg = f"Lean verification timed out after {LEAN_TIMEOUT} seconds"
            logger.error(error_msg)
            return {
                "success": False,
                "output": f"Error: {error_msg}"
            }
            
        except Exception as e:
            error_msg = f"Unexpected error during Lean verification: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "output": f"Error: {error_msg}"
            }
            
        finally:
            # Cleanup temporary files
            logger.info("Cleaning up temporary files")
            if temp_file_path.exists():
                temp_file_path.unlink()
                logger.debug(f"Removed temporary file: {temp_file_path}")
            if olean_path.exists():
                olean_path.unlink()
                logger.debug(f"Removed temporary .olean file: {olean_path}")


# Create an instance of the tool
verify_lean_proof = VerifyLeanProof()