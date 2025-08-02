#!/usr/bin/env python3
"""
Universal Lean Code Verifier
A tool to verify Lean code correctness by compiling it with lake

Usage:
    python lean_verifier.py <lean_code> [theorem_name]
    
Example:
    python lean_verifier.py "import MiniF2F.Minif2fImport; theorem test : 1 + 1 = 2 := by norm_num" my_theorem
"""

import subprocess
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from config import MINIF2F_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lean_verifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_executable(name: str) -> Optional[str]:
    """
    Find executable in PATH or common locations
    
    Args:
        name: Name of the executable to find
        
    Returns:
        Path to executable or None if not found
    """
    # Check if it's in PATH
    try:
        result = subprocess.run(['which', name], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Check common locations
    common_paths = [
        os.path.expanduser(f"~/.elan/bin/{name}"),  # Elan installation
        f"/usr/local/bin/{name}",  # System installation
        f"/opt/lean/bin/{name}",  # Alternative system installation
        f"/usr/bin/{name}",  # Standard system location
    ]
    
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None

class LeanVerifier:
    """
    Universal system for verifying Lean code correctness
    """
    
    def __init__(self, project_dir: str | None = None):
        """
        Initialize the Lean verifier
        
        Args:
            project_dir: Path to the Lean project directory
        """
        # Use provided project_dir or fallback to configuration's MINIF2F_DIR
        self.project_dir = Path(project_dir) if project_dir is not None else Path(MINIF2F_DIR)
        self._setup_paths()
        
    def _setup_paths(self):
        """Setup Lean and Lake binary paths automatically"""
        try:
            # Find lean binary
            self.lean_bin = find_executable("lean")
            if not self.lean_bin:
                raise RuntimeError("Lean binary not found. Please install Lean 4.")
            
            # Find lake binary
            self.lake_bin = find_executable("lake")
            if not self.lake_bin:
                raise RuntimeError("Lake binary not found. Please install Lean 4.")
                
            logger.info(f"Found Lean binary: {self.lean_bin}")
            logger.info(f"Found Lake binary: {self.lake_bin}")
            
        except Exception as e:
            logger.error(f"Failed to setup Lean paths: {e}")
            raise
        
    def verify_lean_code(self, lean_code: str, theorem_name: str = "test_theorem") -> Dict[str, Any]:
        """
        Verify Lean code correctness by compiling it
        
        Args:
            lean_code: Lean code to verify
            theorem_name: Name for the theorem (used for file naming)
            
        Returns:
            dict with verification results containing:
            - success: bool - whether compilation succeeded
            - output: str - compilation output
            - error: str - error message if compilation failed
        """
        
        logger.info(f"Starting verification of theorem: {theorem_name}")
        
        # Create temporary file in the project
        theorem_file = self.project_dir / "MiniF2F" / f"{theorem_name}.lean"
        
        try:
            # Ensure the directory exists
            theorem_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the theorem code
            logger.debug(f"Writing theorem code to: {theorem_file}")
            with theorem_file.open("w", encoding='utf-8') as f:
                f.write(lean_code)
            
            # Compile with lake
            logger.info("Compiling theorem with lake...")
            result = self._compile_with_lake(theorem_name)
            
            if result["success"]:
                logger.info("✅ Theorem compiled successfully!")
            else:
                logger.warning("❌ Theorem compilation failed!")
                
            return result
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {
                "success": False,
                "error": f"Verification error: {str(e)}",
                "output": None
            }
        finally:
            # Clean up temporary file
            self._cleanup_file(theorem_file)
    
    def _compile_with_lake(self, theorem_name: str) -> Dict[str, Any]:
        """
        Compile theorem using lake build system
        
        Args:
            theorem_name: Name of the theorem to compile
            
        Returns:
            dict with compilation results
        """
        
        # Setup environment
        env = os.environ.copy()
        
        # Add lean/lake to PATH if not already there
        lean_dir = os.path.dirname(self.lean_bin)
        if lean_dir not in env.get('PATH', ''):
            env['PATH'] = f"{lean_dir}:{env.get('PATH', '')}"
        
        try:
            logger.debug(f"Running: lake build MiniF2F.{theorem_name}")
            
            result = subprocess.run(
                [self.lake_bin, "build", f"MiniF2F.{theorem_name}"],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes timeout
                env=env
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Compilation timeout (120 seconds)")
            return {
                "success": False,
                "error": "Compilation timeout (120 seconds)"
            }
        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return {
                "success": False,
                "error": f"Compilation error: {str(e)}"
            }
    
    def _cleanup_file(self, file_path: Path):
        """Clean up temporary file"""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")

def main():
    """Main function - CLI interface"""
    
    if len(sys.argv) < 2:
        print("Lean Code Verifier")
        print("=" * 50)
        print("Usage: python lean_verifier.py <lean_code> [theorem_name]")
        print()
        print("Examples:")
        print('  python lean_verifier.py "import MiniF2F.Minif2fImport; theorem test : 1 + 1 = 2 := by norm_num"')
        print('  python lean_verifier.py "import MiniF2F.Minif2fImport; theorem complex : ∀ x : ℝ, x + 0 = x := by simp" my_theorem')
        print()
        print("The script will:")
        print("  - Compile your Lean code")
        print("  - Report success or detailed error messages")
        print("  - Log all operations to lean_verifier.log")
        sys.exit(1)
    
    lean_code = sys.argv[1]
    theorem_name = sys.argv[2] if len(sys.argv) > 2 else "test_theorem"
    
    try:
        logger.info("Starting Lean Code Verifier")
        verifier = LeanVerifier()
        result = verifier.verify_lean_code(lean_code, theorem_name)
        
        if result["success"]:
            print("✅ Code compiled successfully!")
            if result.get("output"):
                print(f"Output: {result['output']}")
        else:
            print("❌ Compilation failed!")
            if result.get("error"):
                print(f"Error: {result['error']}")
            if result.get("output"):
                print(f"Output: {result['output']}")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        print("Please check that Lean 4 is properly installed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 