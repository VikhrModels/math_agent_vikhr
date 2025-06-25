import sys
import json
import random
import logging
import argparse
from pathlib import Path

# Add the parent directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import CodeAgent, OpenAIServerModel
from config import (
    DEFAULT_MODEL, OPENROUTER_API_BASE, OPENROUTER_API_KEY, 
    MINIF2F_DIR, LOG_DIR, TMP_DIR, LEAN_OUTPUT_FILE,
    DEFAULT_SUBSET_SIZE, DEFAULT_LOG_LEVEL, 
    DEFAULT_MAX_STEPS, DEFAULT_PLANNING_INTERVAL,
    validate_config
)
from agents.tools import verify_lean_proof

# --- Logging setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "agent_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper functions ---
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

    logger.info(f"Selected {len(micro_subset)} theorems for testing")
    return micro_subset

def create_agent(max_steps: int = DEFAULT_MAX_STEPS, planning_interval: int = DEFAULT_PLANNING_INTERVAL):
    """
    Create and configure the math prover agent.
    Args:
        max_steps: Maximum number of agent steps per theorem
        planning_interval: Interval for agent planning steps
    """
    try:
        validate_config()
    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)

    model = OpenAIServerModel(
        model_id=DEFAULT_MODEL,
        api_base=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
    )

    agent = CodeAgent(
        tools=[verify_lean_proof], 
        model=model,
        max_steps=max_steps,
        planning_interval=planning_interval
    )
    
    logger.info(f"Created agent with max_steps={max_steps}, planning_interval={planning_interval}")
    return agent

def prove_theorem_with_agent(agent: CodeAgent, theorem: dict, max_steps: int = DEFAULT_MAX_STEPS) -> bool:
    """
    Uses the agent to prove a single theorem.
    Args:
        agent: The configured math prover agent
        theorem: Dictionary containing theorem data
        max_steps: Maximum number of agent steps per theorem
    Returns:
        bool: True if the theorem was successfully proven, False otherwise
    """
    logger.info(f"\n--- Processing Theorem: {theorem['name']} ---")
    logger.info(f"Original statement (first 200 chars):\n{theorem['statement'][:200]}...")
    logger.info(f"Agent limit: max_steps={max_steps}")

    try:
        prompt = (
            f"You are a Lean 3.42.1 theorem prover. Your task is to generate Lean code that proves the given theorem.\n\n"
            f"CRITICAL: You must ONLY generate Lean code and use the verify_lean_proof tool. DO NOT write any Python code, mathematical analysis, or calculations.\n\n"
            f"WORKFLOW:\n"
            f"1. Look at the theorem statement\n"
            f"2. Periodically (every planning_interval steps), perform a planning phase: generate and update a list of possible proof strategies (hypotheses) for the theorem.\n"
            f"3. For each hypothesis, attempt to generate a complete Lean theorem with proper tactics (replace 'sorry' with Lean tactics like norm_num, simp, rw, exact, etc.)\n"
            f"4. Use verify_lean_proof(theorem_statement) to check your Lean code\n"
            f"5. If verification fails, update your plan and try a new hypothesis or fix the Lean code and try again\n"
            f"6. Return ONLY the final verified Lean theorem\n\n"
            f"LEAN TACTICS YOU CAN USE:\n"
            f"- norm_num: for numerical calculations\n"
            f"- simp: for simplification\n"
            f"- rw: for rewriting with hypotheses\n"
            f"- exact: for exact proofs\n"
            f"- apply: for applying lemmas\n"
            f"- cases: for case analysis\n"
            f"- induction: for induction\n"
            f"- have: for introducing new hypotheses\n"
            f"- proof blocks: use {{ }} for subproofs\n"
            f"- ring: for ring theory\n"
            f"- linarith: for linear arithmetic\n"
            f"- etc.\n\n"
            f"IMPORTANT: Do not use 'sorry' anywhere in your proof, including inside proof blocks.\n\n"
            f"EXAMPLE OF CORRECT APPROACH:\n"
            f"theorem_statement = '''theorem example : 2 + 2 = 4 := begin norm_num end'''\n"
            f"result = verify_lean_proof(theorem_statement)\n\n"
            f"DO NOT DO THIS (WRONG):\n"
            f"# Mathematical analysis in Python\n"
            f"for i in range(10):\n"
            f"    print(i)\n\n"
            f"Your response must be ONLY Lean code starting with 'theorem' and ending with 'end'.\n"
            f"No Python code, no comments, no explanations.\n\n"
            f"Prove this theorem:\n{theorem['statement']}"
        )

        result = agent.run(prompt)
        # Если результат None — неуспех
        if result is None:
            logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (no result)")
            logger.debug(f"Agent result:\n{result}")
            return False
        # Если результат — dict, только success==True — успех, иначе сразу return False
        if isinstance(result, dict):
            if result.get('success') is True:
                logger.info(f"✅ Agent successfully generated proof for {theorem['name']}")
                logger.debug(f"Agent result:\n{result}")
                return True
            else:
                logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (agent returned success: False or error)")
                logger.debug(f"Agent result:\n{result}")
                return False
        # Если результат — строка с Lean-доказательством, проверить его через verify_lean_proof
        if isinstance(result, str) and 'theorem' in result and 'begin' in result and 'end' in result:
            verification = verify_lean_proof(result)
            if isinstance(verification, dict) and verification.get('success') is True:
                logger.info(f"✅ Agent successfully generated proof for {theorem['name']} (after verification)")
                logger.debug(f"Agent result:\n{result}")
                return True
            else:
                logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (verification failed)")
                logger.debug(f"Verification result:\n{verification}")
                return False
        # Всё остальное — неуспех
        logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (no valid result)")
        logger.debug(f"Agent result:\n{result}")
        return False
    except Exception as e:
        logger.error(f"❌ Error during agent execution for {theorem['name']}: {e}")
        return False

def main():
    """
    Main function to run the agent-based theorem proving benchmark.
    """
    parser = argparse.ArgumentParser(description="Agent-based Lean theorem proving benchmark.")
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE,
                        help="Total number of tasks for the micro-subset. Use 0 or -1 to use the full dataset.")
    parser.add_argument("--json_file", type=Path, default=LEAN_OUTPUT_FILE,
                        help="Path to the JSON file containing theorems.")
    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum number of agent steps per theorem.")
    parser.add_argument("--planning_interval", type=int, default=DEFAULT_PLANNING_INTERVAL,
                        help=f"Interval for agent planning steps (default: {DEFAULT_PLANNING_INTERVAL}). Lower = more frequent planning.")
    args = parser.parse_args()

    MICRO_SUBSET_SIZE = args.subset_size
    VALID_JSON_PATH = args.json_file
    MAX_STEPS = args.max_steps
    PLANNING_INTERVAL = args.planning_interval
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Starting Agent-based MiniF2F Lean Prover Benchmark...")
    logger.info(f"Configuration: Subset Size={MICRO_SUBSET_SIZE}, JSON File='{VALID_JSON_PATH}', Model='{DEFAULT_MODEL}', Log Level='{args.log_level}', Max Steps={MAX_STEPS}, Planning Interval={PLANNING_INTERVAL}")
    
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        all_theorems = read_json_file(VALID_JSON_PATH)
    except Exception as e:
        logger.critical(f"Failed to load JSON file: {e}. Exiting.")
        return
    
    if not all_theorems:
        logger.warning("No theorems found in the JSON file. Exiting.")
        return

    if MICRO_SUBSET_SIZE <= 0:
        micro_subset = all_theorems
        logger.info(f"Special value for subset_size={MICRO_SUBSET_SIZE}: using the full dataset ({len(all_theorems)} tasks).")
    else:
        micro_subset = select_micro_subset_stratified(all_theorems, MICRO_SUBSET_SIZE)

    if not micro_subset:
        logger.warning("No tasks selected for the micro-subset. Exiting.")
        return

    agent = create_agent(max_steps=MAX_STEPS, planning_interval=PLANNING_INTERVAL)

    logger.info("\n--- Running Agent-based Theorem Proving on Micro-Subset ---")
    results = {}
    
    for i, theorem in enumerate(micro_subset):
        logger.info(f"\nProcessing task {i+1}/{len(micro_subset)}: {theorem['name']}")
        success = prove_theorem_with_agent(agent, theorem, max_steps=MAX_STEPS)
        results[theorem['name']] = success

    logger.info("\n--- Summary of Results ---")
    solved_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")

    logger.info(f"\nTotal tasks processed: {total_count}")
    logger.info(f"Successfully proven: {solved_count}")
    
    if total_count > 0:
        success_rate = solved_count / total_count * 100
        logger.info(f"Pass rate: {success_rate:.2f}%")
    else:
        logger.info("No tasks processed.")

if __name__ == "__main__":
    main()