import os
import re
import subprocess
import time
import json
import random
import logging
import argparse
from pathlib import Path
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log

# Import configuration
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    DEFAULT_MODEL,
    MINIF2F_DIR,
    LOG_DIR,
    TMP_DIR,
    LEAN_OUTPUT_FILE,
    LEAN_TIMEOUT,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_JSON_FILE,
    LOG_FORMAT,
    validate_config
)

# Import agents
from agents import create_math_prover_agent

# --- Constants and default configuration ---
# This section defines constants and default settings for the script.
# These can be customized as needed.

# Path to Lean 4 executable. Make sure 'lean' is available in PATH,
# or specify the full path, e.g., '/opt/lean4/bin/lean'
LEAN_EXECUTABLE_PATH = "lean"

# Micro-subset file for tracking selected tasks
MICRO_SUBSET_FILE = Path(__file__).parent / "micro_subset.txt"

# --- Logging setup ---
# Configures a robust logging system to track the script's execution,
# writing logs to both a file and the console.
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / "llm_requests.log"),
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)

# --- OpenRouter client initialization ---
# Sets up the API client for communication with the OpenRouter service.
# Exits gracefully if the required API key is not found.
try:
    validate_config()
    client = OpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
    )
except (ValueError, FileNotFoundError) as e:
    logger.critical(e)
    exit(1) # Exit if configuration is invalid

# --- Helper functions ---
# This section contains reusable functions that encapsulate specific logic,
# such as file handling, data selection, and Lean proof verification.

def read_json_file(filepath: Path) -> list[dict]:
    """Reads the contents of a JSON file with theorems."""
    logger.info(f"Reading JSON file: {filepath}")
    try:
        with filepath.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

def select_micro_subset_stratified(all_theorems: list[dict], subset_size: int) -> list[dict]:
    """
    Selects a micro-subset of tasks while preserving the ratio of solved/unsolved tasks.
    This ensures that testing subsets are representative of the larger dataset.
    """
    solved_theorems = [t for t in all_theorems if t['is_solved']]
    unsolved_theorems = [t for t in all_theorems if not t['is_solved']]

    total_solved = len(solved_theorems)
    total_unsolved = len(unsolved_theorems)
    total_all = total_solved + total_unsolved

    if total_all == 0:
        logger.warning("No theorems found to select from.")
        return []

    # Calculate target number of tasks for each category
    if total_solved > 0:
        num_solved_in_subset = round(subset_size * (total_solved / total_all))
    else:
        num_solved_in_subset = 0
    
    num_unsolved_in_subset = subset_size - num_solved_in_subset

    logger.info(f"Selecting micro-subset of {subset_size} tasks:")
    logger.info(f"  Total solved theorems: {total_solved}, Target for subset: {num_solved_in_subset}")
    logger.info(f"  Total unsolved theorems: {total_unsolved}, Target for subset: {num_unsolved_in_subset}")

    # Random selection from each category
    selected_solved = random.sample(solved_theorems, min(num_solved_in_subset, total_solved))
    selected_unsolved = random.sample(unsolved_theorems, min(num_unsolved_in_subset, total_unsolved))
    
    micro_subset = selected_solved + selected_unsolved
    random.shuffle(micro_subset)  # Shuffle for random order

    # Save selected tasks to file
    MICRO_SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MICRO_SUBSET_FILE.open('w', encoding='utf-8') as f:
        f.write(f"Micro-subset of {len(micro_subset)} tasks selected from {len(all_theorems)} total:\n")
        f.write(f"  Solved tasks: {len(selected_solved)}\n")
        f.write(f"  Unsolved tasks: {len(selected_unsolved)}\n\n")
        for theorem in micro_subset:
            f.write(f"- {theorem['name']} (Is solved: {theorem['is_solved']})\n")
    logger.info(f"Micro-subset details saved to {MICRO_SUBSET_FILE}")
    
    return micro_subset

def verify_lean_proof(theorem_statement: str, generated_proof_body: str) -> bool:
    """
    Verifies a generated Lean proof by creating a temporary Lean file and compiling it.

    This function simulates the actual environment of the miniF2F project to ensure
    that the generated proof is valid within the project's context, including all
    necessary imports and dependencies.
    """
    logger.info("Attempting Lean 3 verification.")

    if not generated_proof_body.strip():
        logger.warning("  ❌ Empty proof body provided.")
        return False

    # Find the part of the statement before the proof begins.
    # The statements in valid.json end with `:= sorry` or `:= begin sorry end`.
    statement_base_match = re.search(r'(.*):=\s*(begin\s*sorry\s*end|sorry)', theorem_statement, re.DOTALL)
    if not statement_base_match:
        logger.error("  ❌ Could not find `:= sorry` or `:= begin sorry end` in the theorem statement.")
        logger.debug(f"    Full statement: {theorem_statement}")
        return False
    statement_base = statement_base_match.group(1).strip()

    # Create complete theorem with generated proof
    full_lean_code = statement_base + " :=\n" + generated_proof_body

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
            f.write(full_lean_code)

        # We run `lean --make` from the root of the miniF2F project.
        # This command compiles the file and its dependencies, which is the
        # standard way to build a Lean project file.
        process = subprocess.run(
            [LEAN_EXECUTABLE_PATH, "--make", f"lean/src/{temp_file_name}"],
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
            logger.warning(f"  ❌ Verification failed for {theorem_statement.splitlines()[0]}: Lean reported errors.")
            # Extract and log only error lines for clarity
            error_lines = [line for line in output.splitlines() if "error:" in line.lower()]
            if error_lines:
                logger.warning("  Lean errors:")
                for error in error_lines:
                    logger.warning(f"    {error.strip()}")
            return False

        # For `lean --make`, a successful compilation often produces no output.
        # The presence of the .olean file and no errors is a good indicator of success.
        if not olean_path.exists():
             logger.warning(f"  ❌ Verification failed: .olean file was not produced, but no errors were reported.")
             return False

        logger.info(f"  ✅ Verification successful for {theorem_statement.splitlines()[0]}.")
        return True

    except FileNotFoundError:
        logger.error(f"  ❌ Lean executable '{LEAN_EXECUTABLE_PATH}' not found. Make sure Lean 3 is installed and in your PATH.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"  ❌ Lean verification timed out for {temp_file_path.name} after {LEAN_TIMEOUT} seconds.")
        return False
    except Exception as e:
        logger.error(f"  ❌ An unexpected error occurred during Lean verification for {temp_file_path.name}: {e}")
        return False
    finally:
        # Cleanup the temporary files
        if temp_file_path.exists():
            temp_file_path.unlink()
        if olean_path.exists():
            olean_path.unlink()

def extract_proof_body(llm_response_content: str) -> str | None:
    """
    Extracts the Lean proof body from the LLM's response.

    The function searches for a Lean code block in markdown and then validates
    if the content is a standard proof format (a `begin...end` block or a `by` tactic).
    """
    # First, try to find Lean code in markdown blocks
    match = re.search(r'```(?:lean)?\s*(.*?)\s*```', llm_response_content, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
    else:
        # If no markdown, use the whole response as content, stripped of leading/trailing whitespace
        content = llm_response_content.strip()

    # The proof must be a self-contained block.
    # Check for 'begin...end' or 'by ...'
    # Use re.DOTALL to match newlines within begin...end
    if re.match(r'begin.*end', content, re.DOTALL | re.IGNORECASE):
        logger.info("  Extracted proof body from a `begin...end` block.")
        return content
    
    if content.lower().startswith('by '):
        logger.info("  Extracted proof body from a `by` expression.")
        return content

    logger.warning("  Failed to extract a valid Lean proof body (a `begin...end` block or `by ...`) from LLM response.")
    logger.debug(f"    Full response content:\n{llm_response_content}")
    return None

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5),
       before_sleep=before_sleep_log(logger, logging.WARNING))
def _call_llm_with_retry(messages: list[dict], model_name: str, extra_headers: dict, max_tokens: int, temperature: float):
    """
    A wrapper for the OpenAI API call that includes automatic retries with exponential backoff.
    This makes the script more resilient to transient network issues or API rate limits.
    """
    return client.chat.completions.create(
        extra_headers=extra_headers,
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def generate_and_verify_proof(theorem: dict) -> bool:
    """
    Orchestrates the full process for a single theorem:
    1. Constructs a prompt for the LLM.
    2. Sends the request to the LLM to generate a proof.
    3. Extracts the proof from the response.
    4. Verifies the proof using the Lean executable.
    """
    logger.info(f"\n--- Processing Theorem: {theorem['name']} ---")
    logger.info(f"Original statement (first 200 chars):\n{theorem['statement'][:200]}...")

    try:
        system_prompt = (
            "You are a highly qualified Lean 3.42.1 theorem prover. "
            "Your task is to generate valid, formal proofs for mathematical theorems in Lean 3.42.1. "
            "You will be given a theorem with a `sorry` placeholder. You must replace the `sorry` with a complete proof. "
            "Your response must be ONLY the Lean code for the proof, starting with `begin` and ending with `end`, or starting with `by`. "
            "The code should be wrapped in a markdown block with the language specifier `lean` (e.g., ```lean ... ```). "
            "Do not include any additional explanations, comments, introductions, or conclusions."
        )

        user_prompt = (
            f"Prove the following theorem in Lean 3.42.1 by replacing `sorry` with a full proof. "
            "Your response should be only the proof block (e.g., `begin ... end` or `by ...`), "
            "wrapped in a Lean code block (```lean ... ```).\n\n"
            f"```lean\n{theorem['statement']}\n```"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.info(f"Sending request to LLM for {theorem['name']}. Prompt (first 500 chars):\n{json.dumps(messages, indent=2)[:500]}...")

        completion = _call_llm_with_retry(
            messages=messages,
            model_name=DEFAULT_MODEL,
            extra_headers={
                "HTTP-Referer": "https://github.com/umbra2728/math_agent_vikhr",
                "X-Title": "Math Agent Vikhr",
            },
            max_tokens=2048,
            temperature=0.1,
        )

        generated_content = completion.choices[0].message.content
        
        logger.info(f"Received LLM response for {theorem['name']}. Content :\n{generated_content[:]}...")

        generated_proof_body = extract_proof_body(generated_content)
        if generated_proof_body:
            logger.info(f"  Attempting to verify proof for {theorem['name']}...")
            return verify_lean_proof(theorem['statement'], generated_proof_body)
        else:
            logger.warning(f"  ❌ Failed to extract Lean proof body from LLM response for {theorem['name']}. Skipping verification.")
            return False

    except Exception as e:
        logger.error(f"  An error occurred during LLM interaction for {theorem['name']}: {e}")
        return False

# --- Main logic ---
def main():
    """
    Main function to run the script.

    Parses command-line arguments, sets up the environment, reads theorems,
    runs the proof generation and verification loop, and prints a summary of the results.
    """
    global DEFAULT_MODEL

    parser = argparse.ArgumentParser(description="Automated Lean 4 theorem proving with LLM.")
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE,
                        help="Total number of tasks for the micro-subset.")
    parser.add_argument("--json_file", type=Path, default=LEAN_OUTPUT_FILE,
                        help="Path to the JSON file containing theorems.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="OpenRouter model name to use for proof generation.")
    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (e.g., DEBUG for verbose output).")
    args = parser.parse_args()

    MICRO_SUBSET_SIZE = args.subset_size
    VALID_JSON_PATH = args.json_file
    DEFAULT_MODEL = args.model
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Starting MiniF2F Lean Prover Benchmark with Claude Sonnet via OpenRouter...")
    logger.info(f"Configuration: Subset Size={MICRO_SUBSET_SIZE}, JSON File='{VALID_JSON_PATH}', Model='{DEFAULT_MODEL}', Log Level='{args.log_level}'")
    
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        all_theorems = read_json_file(VALID_JSON_PATH)
    except Exception as e:
        logger.critical(f"Failed to load JSON file: {e}. Exiting.")
        return
    
    if not all_theorems:
        logger.warning("No theorems found in the JSON file. Exiting.")
        return

    micro_subset = select_micro_subset_stratified(all_theorems, MICRO_SUBSET_SIZE)

    if not micro_subset:
        logger.warning("No tasks selected for the micro-subset. Exiting.")
        return

    logger.info("\n--- Running LLM and Lean Verification on Micro-Subset ---")
    results = {}
    for i, theorem in enumerate(micro_subset):
        logger.info(f"\nProcessing task {i+1}/{len(micro_subset)}: {theorem['name']}")
        success = generate_and_verify_proof(theorem)
        results[theorem['name']] = success
        time.sleep(1)

    logger.info("\n--- Summary of Results ---")
    solved_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")

    logger.info(f"\nTotal tasks processed: {total_count}")
    logger.info(f"Successfully proven: {solved_count}")
    
    if total_count > 0:
        logger.info(f"Pass rate: {solved_count / total_count * 100:.2f}%")
    else:
        logger.info("No tasks processed.")

if __name__ == "__main__":
    main()