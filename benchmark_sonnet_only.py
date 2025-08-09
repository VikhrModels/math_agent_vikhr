import os
import re
import subprocess
import time
import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log
import concurrent.futures
from agents.tools import lean_verifier as verifier

# Global LLM client reference (initialized in main)
client: Optional[OpenAI] = None
current_provider: Optional[str] = None

# --- Helpers ---------------------------------------------------------------
def _message_text(message) -> str:
    """Extract text content from OpenAI ChatCompletionMessage.

    Handles both string content and list-of-parts responses.
    """
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    # Some SDK versions return a list of content parts
    parts_text: list[str] = []
    if isinstance(content, list):
        for part in content:
            # dict-like {"type": "text", "text": "..."}
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts_text.append(part["text"]) 
            else:
                # object with .type/.text attributes
                text_val = getattr(part, "text", None)
                if isinstance(text_val, str):
                    parts_text.append(text_val)
    return "\n".join(parts_text).strip()

# Import configuration
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    DEFAULT_PROVIDER,
    AVAILABLE_PROVIDERS,
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
    LLM_REQUEST_TIMEOUT,
    validate_config
)

# --- Constants and default configuration ---
# This section defines constants and default settings for the script.
# These can be customized as needed.

# Initialize a single LeanVerifier instance (Lean 4 + Lake)
# verifier = LeanVerifier() # This line is removed as per the edit hint.

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

# --- LLM client factory ---
def make_client(provider: str) -> OpenAI:
    """Create an OpenAI-compatible client for the selected provider."""
    validate_config()
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
        return OpenAI(base_url=OPENROUTER_API_BASE, api_key=OPENROUTER_API_KEY)
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        return OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
    raise ValueError(f"Unsupported provider: {provider}")

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
    If subset_size <= 0, returns the full dataset.
    """
    if subset_size <= 0:
        logger.info(f"Special value for subset_size={subset_size}: using the full dataset ({len(all_theorems)} tasks).")
        return all_theorems
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

def verify_lean_proof(theorem_name: str, theorem_statement: str, generated_proof_body: str) -> bool:
    """
    Verifies a generated Lean proof by creating a temporary Lean file and compiling it.

    This function simulates the actual environment of the miniF2F project to ensure
    that the generated proof is valid within the project's context, including all
    necessary imports and dependencies.
    """
    logger.info("Attempting Lean 4 verification via Lake.")

    if not generated_proof_body.strip():
        logger.warning("  ❌ Empty proof body provided.")
        return False

    # Split off the original (unsolved) proof placeholder after ':='
    if ':=' not in theorem_statement:
        logger.error("  ❌ Expected ':=' in theorem statement but not found.")
        logger.debug(f"    Full statement: {theorem_statement}")
        return False
    statement_base = theorem_statement.split(':=')[0].strip()

    # Create complete theorem with generated proof
    full_lean_code = statement_base + " :=\n" + generated_proof_body

    # Prepare full Lean code with necessary imports for Lean 4 project
    lean_imports = "import MiniF2F.Minif2fImport\n\n"
    lean_code = lean_imports + full_lean_code

    try:
        result = verifier.verify_lean_code(lean_code, theorem_name)
        if result["success"]:
            logger.info(f"  ✅ Verification successful for {theorem_name}.")
            return True
        else:
            logger.warning(f"  ❌ Verification failed for {theorem_name}.")
            if result.get("error"):
                logger.debug(f"    Error: {result['error']}")
            if result.get("output"):
                logger.debug(f"    Output: {result['output']}")
            return False
    except Exception as e:
        logger.error(f"  ❌ An unexpected error occurred during Lean verification for {theorem_name}: {e}")
        return False

def extract_proof_body(llm_response_content: str) -> str | None:
    """
    Extracts the Lean proof body from the LLM's response.

    The function searches for a Lean code block in markdown, then extracts
    the `begin...end` block or a `by` tactic. It handles nested blocks
    and cases where the model returns the full theorem.
    """
    # First, try to find Lean code in markdown blocks
    markdown_match = re.search(r'```(?:lean)?\s*(.*?)\s*```', llm_response_content, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        content = markdown_match.group(1).strip()
    else:
        # If no markdown, use the whole response, stripped of leading/trailing whitespace
        content = llm_response_content.strip()

    # More robustly find the start of the proof, which can be `begin` or `by`
    # This handles cases where the full theorem is repeated in the response.
    begin_match = re.search(r'\b(begin|by\b)', content, re.IGNORECASE | re.DOTALL)
    
    if not begin_match:
        logger.warning("  Failed to find 'begin' or 'by' keyword in the response content.")
        logger.debug(f"    Full response content:\n{llm_response_content}")
        return None
        
    keyword = begin_match.group(1).lower()
    start_index = begin_match.start()
    
    # Slice content from the keyword onward
    proof_search_area = content[start_index:]

    if keyword == 'by':
        # For 'by' expressions, we need to capture the entire proof
        # Look for the end of the proof (usually a newline or end of content)
        # But be careful not to include the rest of the theorem if it's repeated
        
        # Find where the proof ends - either at end of content or before another theorem declaration
        lines = proof_search_area.split('\n')
        proof_lines = []
        
        for line in lines:
            line = line.strip()
            # Stop if we hit another theorem declaration or empty line followed by theorem-like content
            if (line.startswith('theorem ') or 
                line.startswith('lemma ') or 
                line.startswith('def ') or
                (line == '' and any(l.strip().startswith('theorem ') for l in lines[lines.index(line)+1:lines.index(line)+3]))):
                break
            if line:  # Only add non-empty lines
                proof_lines.append(line)
        
        proof_body = '\n  '.join(proof_lines)
        
        # Check if the proof looks complete (not truncated)
        if proof_body and not proof_body.strip().endswith('...'):
            logger.info("  Extracted proof body from a `by` expression.")
            return proof_body
        else:
            logger.warning("  ❌ Extracted proof appears to be incomplete or truncated.")
            logger.debug(f"    Extracted proof: {proof_body}")
            return None

    if keyword == 'begin':
        # Find matching 'end' for the 'begin' block
        open_blocks = 0
        current_pos = 0
        pattern = re.compile(r'\b(begin|end)\b', re.IGNORECASE)
        
        for match in pattern.finditer(proof_search_area):
            if match.group(1).lower() == 'begin':
                open_blocks += 1
            elif match.group(1).lower() == 'end':
                open_blocks -= 1
            
            if open_blocks == 0:
                end_pos = match.end()
                proof_body = proof_search_area[:end_pos]
                logger.info("  Extracted proof body from a `begin...end` block.")
                return proof_body

    logger.warning("  Failed to extract a valid Lean proof body (a `begin...end` block or `by ...`) from LLM response.")
    logger.debug(f"    Full response content:\n{llm_response_content}")
    return None

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5),
       before_sleep=before_sleep_log(logger, logging.WARNING))
def _call_llm_with_retry(messages: List[ChatCompletionMessageParam], model_name: str, extra_headers: dict, max_tokens: int, temperature: float):
    """
    A wrapper for the OpenAI API call that includes automatic retries with exponential backoff.
    This makes the script more resilient to transient network issues or API rate limits.
    """
    # Use provider-specific token parameter to avoid 400 errors on OpenAI (gpt-5, etc.)
    if current_provider == "openai":
        # For OpenAI (e.g., gpt-5), do not send temperature (only default=1 supported)
        # and use max_completion_tokens instead of max_tokens
        return client.chat.completions.create(
            extra_headers=extra_headers,
            model=model_name,
            messages=messages,
            extra_body={"max_completion_tokens": max_tokens},
            timeout=LLM_REQUEST_TIMEOUT,
        )
    # Default: OpenRouter
    return client.chat.completions.create(
        extra_headers=extra_headers,
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=LLM_REQUEST_TIMEOUT,
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
            "You are a Lean 4 theorem prover. Your task is to replace `sorry` with a complete proof. "
            "Generate ONLY the proof code, no explanations or comments. "
            "Use simple tactics: `simp`, `norm_num`, `ring`, `linarith`, `tauto`, `exact`, `apply`, `rw`. "
            "For simple calculations use `norm_num` or `ring`. "
            "For linear equations use `linarith`. "
            "For rewriting use `rw [h₀, h₁]` then `ring` or `linarith`. "
            "Response must be ONLY the proof code in ```lean``` block. "
            "Keep proofs short and direct."
        )

        user_prompt = (
            f"Replace `sorry` with a simple Lean 4 proof using only basic tactics (simp, norm_num, ring, linarith, tauto, exact, apply, rw, cases, induction). Return ONLY the proof code in ```lean``` block.\n\n"
            f"Examples of simple proofs:\n"
            f"```lean\n"
            f"theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by ring_nf\n"
            f"```\n"
            f"```lean\n"
            f"theorem mathd_algebra_462 : ((1 : ℚ) / 2 + 1 / 3) * (1 / 2 - 1 / 3) = 5 / 36 := by norm_num\n"
            f"```\n"
            f"```lean\n"
            f"theorem mathd_numbertheory_132 : 2004 % 12 = 0 := by norm_num\n"
            f"```\n"
            f"```lean\n"
            f"theorem mathd_algebra_455 (x : ℝ) (h₀ : 2 * (2 * (2 * (2 * x))) = 48) : x = 3 := by linarith\n"
            f"```\n"
            f"```lean\n"
            f"theorem amc12a_2008_p2 (x : ℝ) (h₀ : x * (1 / 2 + 2 / 3) = 1) : x = 6 / 7 := by linarith\n"
            f"```\n"
            f"```lean\n"
            f"theorem mathd_algebra_48 (q e : ℂ) (h₀ : q = 9 - 4 * Complex.I) (h₁ : e = -3 - 4 * Complex.I) : q - e = 12 := by\n"
            f"  rw [h₀, h₁]\n"
            f"  ring\n"
            f"```\n\n"
            f"Now prove this theorem:\n\n"
            f"```lean\n{theorem['statement']}\n```"
        )

        messages: List[ChatCompletionMessageParam] = [
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
            max_tokens=4096,
            temperature=0.1,
        )

        # Check if completion and choices exist
        if not completion or not completion.choices:
            logger.error(f"  ❌ No response received from LLM for {theorem['name']}")
            return False
        
        # Check if message and content exist (handle both string and content parts)
        message = completion.choices[0].message
        generated_content = _message_text(message)
        if not generated_content:
            logger.error(f"  ❌ Empty response content from LLM for {theorem['name']}")
            return False
        finish_reason = completion.choices[0].finish_reason
        logger.info(f"LLM response for {theorem['name']} finished with reason: {finish_reason}.")
        
        logger.info(f"Received LLM response for {theorem['name']}. Content:\n{generated_content[:200]}...")

        generated_proof_body = extract_proof_body(generated_content)
        if generated_proof_body:
            logger.info(f"  Attempting to verify proof for {theorem['name']}...")
            return verify_lean_proof(theorem['name'], theorem['statement'], generated_proof_body)
        else:
            logger.warning(f"  ❌ Failed to extract Lean proof body from LLM response for {theorem['name']}. Skipping verification.")
            return False

    except Exception as e:
        logger.error(f"  An error occurred during LLM interaction for {theorem['name']}: {e}")
        return False

def process_theorem_task(theorem: dict) -> tuple[bool, str]:
    """
    Wrapper for thread pool: runs generate_and_verify_proof and returns (success, theorem_name)
    """
    success = generate_and_verify_proof(theorem)
    return success, theorem['name']

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
                        help="Total number of tasks for the micro-subset. Use 0 or -1 to use the full dataset.")
    parser.add_argument("--json_file", type=Path, default=LEAN_OUTPUT_FILE,
                        help="Path to the JSON file containing theorems.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name to use for proof generation.")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER,
                        choices=AVAILABLE_PROVIDERS,
                        help=f"LLM provider (default: {DEFAULT_PROVIDER}). Choices: {', '.join(AVAILABLE_PROVIDERS)}")
    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (e.g., DEBUG for verbose output).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of theorems to process in parallel (default: 4).")
    args = parser.parse_args()

    MICRO_SUBSET_SIZE = args.subset_size
    VALID_JSON_PATH = args.json_file
    DEFAULT_MODEL = args.model
    provider = args.provider
    CONCURRENCY = args.concurrency
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Starting MiniF2F Lean Prover Benchmark...")
    logger.info(f"Configuration: Subset Size={MICRO_SUBSET_SIZE}, JSON File='{VALID_JSON_PATH}', Provider='{provider}', Model='{DEFAULT_MODEL}', Log Level='{args.log_level}', Concurrency={CONCURRENCY}")
    
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        all_theorems = read_json_file(VALID_JSON_PATH)
    except Exception as e:
        logger.critical(f"Failed to load JSON file: {e}. Exiting.")
        return
    
    if not all_theorems:
        logger.warning("No theorems found in the JSON file. Exiting.")
        return

    # Special value: use full dataset if subset_size <= 0
    if MICRO_SUBSET_SIZE <= 0:
        micro_subset = all_theorems
        logger.info(f"Special value for subset_size={MICRO_SUBSET_SIZE}: using the full dataset ({len(all_theorems)} tasks).")
    else:
        micro_subset = select_micro_subset_stratified(all_theorems, MICRO_SUBSET_SIZE)

    if not micro_subset:
        logger.warning("No tasks selected for the micro-subset. Exiting.")
        return

    logger.info("\n--- Running LLM and Lean Verification on Micro-Subset (multithreaded) ---")
    global client
    client = make_client(provider)
    global current_provider
    current_provider = provider
    results = {}
    # Limit to one worker for OpenAI to simplify rate-limits/hangs; keep concurrency for OpenRouter
    max_workers = 1 if provider == "openai" else CONCURRENCY
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_theorem = {
            executor.submit(process_theorem_task, theorem): theorem['name']
            for theorem in micro_subset
        }
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_theorem):
            theorem_name = future_to_theorem[future]
            try:
                success, name = future.result(timeout=360) # 6 min timeout per theorem
                results[name] = success
            except concurrent.futures.TimeoutError:
                logger.error(f"❌ Timeout processing theorem {theorem_name}")
                results[theorem_name] = False
            except Exception as e:
                logger.error(f"❌ Error processing theorem {theorem_name}: {e}")
                results[theorem_name] = False
            processed_count += 1
            logger.info(f"Progress: {processed_count}/{len(micro_subset)} total tasks processed.")

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