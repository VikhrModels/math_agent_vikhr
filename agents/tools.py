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
    
    inputs = {
        "theorem_statement": {
            "type": "string",
            "description": "The complete Lean theorem statement to verify. This should be a full theorem definition that may contain `:= sorry` or `:= begin sorry end`."
        }
    }
    
    output_type = "object"
    
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
        
        # Check if the proof contains 'sorry' - this means it's incomplete
        if "sorry" in theorem_statement.lower():
            # Check if sorry is the only content between begin and end
            begin_end_match = re.search(r'begin\s*(.*?)\s*end', theorem_statement, re.DOTALL | re.IGNORECASE)
            if begin_end_match:
                proof_content = begin_end_match.group(1).strip().lower()
                # If the only content is 'sorry' (with possible whitespace), then it's incomplete
                if proof_content == 'sorry':
                    logger.warning("Proof contains only 'sorry' - marking as incomplete")
                    return {
                        "success": False,
                        "output": "Error: Proof contains only 'sorry' and is incomplete"
                    }
                else:
                    logger.info("Proof contains 'sorry' but also other tactics - proceeding with verification")
            else:
                # If we can't find begin/end structure, check if sorry is the only content
                if theorem_statement.lower().strip() == 'sorry':
                    logger.warning("Theorem contains only 'sorry' - marking as incomplete")
                    return {
                        "success": False,
                        "output": "Error: Theorem contains only 'sorry' and is incomplete"
                    }
        
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
                f.write(theorem_statement)
            
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
            
            # Clean up the output by removing temporary file paths
            cleaned_output = self._clean_output(output, temp_file_name)
            
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
                    "output": cleaned_output
                }
            
            # Check if .olean file was produced (indicator of successful compilation)
            if not olean_path.exists():
                logger.warning("Compilation appeared successful but .olean file was not produced")
                return {
                    "success": False,
                    "output": cleaned_output + "\nWarning: .olean file was not produced despite no errors"
                }
            
            logger.info("Lean compilation successful!")
            return {
                "success": True,
                "output": cleaned_output
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
    
    def _clean_output(self, output: str, temp_file_name: str) -> str:
        """
        Clean up the Lean output by removing temporary file paths and formatting errors properly.
        
        Args:
            output: Raw output from Lean compiler
            temp_file_name: Name of the temporary file to remove from output
            
        Returns:
            Cleaned output string
        """
        # Remove the temporary file path from all lines
        lines = output.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Remove the full path to the temporary file, keeping only line:column info
            if temp_file_name in line:
                # Extract line:column:message format
                parts = line.split(temp_file_name)
                if len(parts) >= 2:
                    # Keep everything after the filename (line:column:message)
                    cleaned_line = parts[1].lstrip(':')
                    cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


# Create an instance of the tool
verify_lean_proof = VerifyLeanProof()