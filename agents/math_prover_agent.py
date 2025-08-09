import sys
import json
import random
import logging
import argparse
from pathlib import Path
import tiktoken
from typing import Union, Optional
import concurrent.futures

# Add the parent directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import CodeAgent, OpenAIServerModel
from config import (
    DEFAULT_MODEL, AVAILABLE_MODELS, OPENROUTER_API_BASE, OPENROUTER_API_KEY, 
    MINIF2F_DIR, LOG_DIR, TMP_DIR, LEAN_OUTPUT_FILE,
    DEFAULT_SUBSET_SIZE, DEFAULT_LOG_LEVEL, 
    DEFAULT_MAX_STEPS, DEFAULT_PLANNING_INTERVAL,
    DEFAULT_CONCURRENCY,
    validate_config
)
from agents.tools import verify_lean_proof, moogle_semantic_search

# --- Logging setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "agent_benchmark.log"),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration even if already configured
)
logger = logging.getLogger(__name__)

# --- Checkpoint configuration ---
CHECKPOINT_DIR = TMP_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

def save_checkpoint(results: dict, processed_count: int, total_count: int, checkpoint_name: str, model_id: str = DEFAULT_MODEL) -> None:
    """Save current progress to a checkpoint file."""
    checkpoint_data = {
        'results': results,
        'processed_count': processed_count,
        'total_count': total_count,
        'timestamp': str(Path().cwd()),
        'model': model_id,
        'subset_size': total_count
    }
    
    checkpoint_file = CHECKPOINT_DIR / f"{checkpoint_name}.json"
    try:
        with checkpoint_file.open('w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_name: str) -> Optional[dict]:
    """Load progress from a checkpoint file."""
    checkpoint_file = CHECKPOINT_DIR / f"{checkpoint_name}.json"
    
    if not checkpoint_file.exists():
        logger.warning(f"Checkpoint file not found: {checkpoint_file}")
        return None
    
    try:
        with checkpoint_file.open('r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        logger.info(f"✅ Checkpoint loaded: {checkpoint_file}")
        logger.info(f"   Processed: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} tasks")
        return checkpoint_data
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return None

def list_checkpoints() -> list[str]:
    """List all available checkpoint files."""
    checkpoints = []
    for checkpoint_file in CHECKPOINT_DIR.glob("*.json"):
        checkpoints.append(checkpoint_file.stem)
    return sorted(checkpoints)

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

def create_math_prover_agent(max_steps: int = DEFAULT_MAX_STEPS, planning_interval: int = DEFAULT_PLANNING_INTERVAL, model_id: str = DEFAULT_MODEL):
    """
    Create a multi-agent system for theorem proving:
    - Idea generator agent: receives a theorem statement, searches for lemmas, forms a proof strategy, and calls the code generator agent.
    - Code generator agent: generates Lean code from the strategy and calls the Lean compiler.
    Returns the main agent (idea generator).
    """
    try:
        validate_config()
    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)

    model = OpenAIServerModel(
        model_id=model_id,
        api_base=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
    )

    # Code generator agent (uses only the Lean proof verifier)
    code_agent = CodeAgent(
        tools=[verify_lean_proof],
        model=model,
        max_steps=10,
        planning_interval=1,
        name="code_generator",
        description=(
            "Generates Lean 4 code from a proof strategy and calls the Lean compiler via Lake. "
            "Returns the code and compiler output. "
            "CRITICAL RULES: "
            "1. NEVER write or execute Python code. "
            "2. NEVER do mathematical calculations in Python. "
            "3. NEVER create variables, functions, or loops in Python. "
            "4. ONLY output valid Lean 4 code. "
            "5. Your output must be a complete Lean theorem statement and proof. "
            "6. EXAMPLE: theorem example : 2 + 2 = 4 := by norm_num. "
            "7. Do NOT include any Python code, comments, or analysis - only Lean code. "
            "8. IMPORTANT: The proof MUST be formatted using 'by' syntax for simple proofs or 'begin' and 'end' syntax for complex proofs. "
            "9. ALWAYS include 'import MiniF2F.Minif2fImport' at the top. "
            "10. Use correct Lean 4 syntax: 'Finset.range' (capital F), not 'finset.range'."
        )
    )

    # Idea generator agent (can search for lemmas and call the code generator)
    idea_agent = CodeAgent(
        tools=[moogle_semantic_search],
        model=model,
        max_steps=10,
        planning_interval=1,
        managed_agents=[code_agent],
        name="idea_generator",
        description="Receives a theorem statement, searches for lemmas, forms a proof strategy, and calls the code generator agent."
    )


    return idea_agent

def prove_theorem_with_agent(agent: CodeAgent, theorem: dict, max_steps: int = DEFAULT_MAX_STEPS, enc=None, token_counter=None) -> Union[bool, str]:
    """
    Uses the agent to prove a single theorem.
    Args:
        agent: The configured math prover agent
        theorem: Dictionary containing theorem data
        max_steps: Maximum number of agent steps per theorem
        enc: tiktoken encoder (optional)
        token_counter: dict with key 'total' to accumulate tokens (optional)
    Returns:
        bool: True if the theorem was successfully proven, False otherwise
    """
    logger.info(f"\n--- Processing Theorem: {theorem['name']} ---")
    logger.info(f"Original statement (first 200 chars):\n{theorem['statement'][:200]}...")
    logger.info(f"Agent limit: max_steps={max_steps}")

    try:
        prompt = (
            f"You are an expert Lean 4 theorem prover agent. You will be given a mathematical theorem to prove.\n"
            f"You have access to the following tools, which you can call as Python functions in your code blocks.\n"
            f"To solve the task, proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.\n\n"

            f"At each step, in the 'Thought:' sequence, explain your reasoning and which tools you want to use.\n"
            f"Then, in the 'Code:' sequence, write the code in simple Python, ending with '<end_code>'.\n"
            f"Use print() to display important intermediate results. These will appear in the 'Observation:' field for the next step.\n"
            f"In the end, return a final answer using the final_answer tool.\n\n"

            f"When searching for relevant lemmas or theorems, batch all necessary moogle_semantic_search queries into a single code block for efficiency. Do not call moogle_semantic_search one at a time in separate steps.\n\n"

            f"EXAMPLE OF BATCHED SEARCHES (CORRECT):\n"
            f"Code:\n"
            f"""```py\nresults_lemma1 = moogle_semantic_search(query=\"lemma about real addition\")\nresults_lemma2 = moogle_semantic_search(query=\"lemma about norm_num\")\nprint(results_lemma1)\nprint(results_lemma2)\n```<end_code>\n"""
            f"Observation: [results for both queries]\n\n"
            f"EXAMPLE OF SINGLE SEARCH PER STEP (WRONG):\n"
            f"Code:\n"
            f"""```py\nresults_lemma1 = moogle_semantic_search(query=\"lemma about real addition\")\nprint(results_lemma1)\n```<end_code>\n"""
            f"Observation: [results for first query]\n\n"
            f"Code:\n"
            f"""```py\nresults_lemma2 = moogle_semantic_search(query=\"lemma about norm_num\")\nprint(results_lemma2)\n```<end_code>\n"""
            f"Observation: [results for second query]\n\n"
            f"Always batch your semantic search queries as shown in the correct example above.\n\n"

            f"While using the code agent you must mention those rules in the prompt:\n"
            f"CRITICAL: You must ONLY generate Lean 4 code and use the verify_lean_proof tool. DO NOT write any Python code, mathematical analysis, or calculations.\n"
            f"EXAMPLE OF CORRECT APPROACH:\n"
            f"theorem_statement = '''theorem example : 2 + 2 = 4 := by norm_num'''\n"
            f"result = verify_lean_proof(theorem_statement)\n"
            f"DO NOT DO THIS (WRONG):\n"
            f"# Mathematical analysis in Python\n"
            f"for i in range(10):\n"
            f"    print(i)\n"
            f"Your response must be ONLY Lean code starting with 'theorem' and ending with 'end'.\n"
            f"No Python code, no comments, no explanations.\n\n"
            f"IMPORTANT: When calling code_generator, NEVER ask it to write Python code or do calculations. "
            f"code_generator should ONLY generate Lean 4 code. If you need mathematical analysis, do it in your own reasoning, "
            f"then pass the strategy to code_generator to convert it to Lean code.\n\n"

            f"Here are a few examples using your available tools and agents:\n"
            f"---\n"
            f"Task: 'Prove a theorem about real number arithmetic.'\n\n"
            f"Thought: I will search for relevant lemmas about real number arithmetic using moogle_semantic_search.\n"
            f"Code:\n"
            f"""```py\nresults = moogle_semantic_search(query=\"real number arithmetic lemma\")\nprint(results)\n```<end_code>\n"""
            f"Observation: [A list of relevant lemmas and their Lean code]\n\n"
            f"Thought: I will now develop a proof strategy and call the code_generator agent to generate and verify Lean code.\n"
            f"Code:\n"
            f"""```py\nstrategy = 'Use the lemma real.add_assoc and norm_num for simplification.'\nlemmas = 'real.add_assoc'\nresult = code_generator(theorem_statement=theorem['statement'], proof_strategy=strategy, lemmas=lemmas)\nprint(result)\n```<end_code>\n"""
            f"Observation: {{'lean_code': '...', 'compiler_output': 'error: ...'}}\n\n"
            f"Thought: The proof failed due to a missing import. I will update my strategy and call code_generator again.\n"
            f"Code:\n"
            f"""```py\nstrategy = 'Add the necessary import for real numbers and use real.add_assoc.'\nlemmas = 'import data.real.basic, real.add_assoc'\nresult = code_generator(theorem_statement=theorem['statement'], proof_strategy=strategy, lemmas=lemmas)\nprint(result)\n```<end_code>\n"""
            f"Observation: {{'lean_code': '...', 'compiler_output': 'success'}}\n\n"
            f"Thought: The proof was successful. I will return the final answer.\n"
            f"Code:\n"
            f"""```py\nfinal_answer(result['lean_code'])\n```<end_code>\n"""
            f"---\n\n"
            f"=== FEW-SHOT EXAMPLES FROM MINIF2F DATASET ===\n"
            f"Here are examples of different complexity levels to guide your proof strategies:\n\n"
            f"1. SIMPLE CALCULATION (norm_num):\n"
            f"theorem mathd_numbertheory_299 : 1 * 3 * 5 * 7 * 9 * 11 * 13 % 10 = 5 := by norm_num\n\n"
            f"2. LINEAR ALGEBRA (linarith):\n"
            f"theorem mathd_algebra_160 (n x : ℝ) (h₀ : n + x = 97) (h₁ : n + 5 * x = 265) : n + 2 * x = 139 := by linarith\n\n"
            f"3. FIELD SIMPLIFICATION + LINEAR:\n"
            f"theorem mathd_algebra_33 (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : 2 * x = 5 * y) (h₂ : 7 * y = 10 * z) : z / x = 7 / 25 := by\n"
            f"  field_simp\n"
            f"  nlinarith\n\n"
            f"4. REWRITING + LINEAR:\n"
            f"theorem mathd_algebra_346 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x - 3) (h₁ : ∀ x, g x = x + 1) : g (f 5 - 1) = 7 := by\n"
            f"  rw [h₀, h₁]\n"
            f"  norm_num\n\n"
            f"5. COMPLEX ALGEBRAIC MANIPULATION:\n"
            f"theorem mathd_algebra_263 (y : ℝ) (h₀ : 0 ≤ 19 + 3 * y) (h₁ : Real.sqrt (19 + 3 * y) = 7) : y = 10 := by\n"
            f"  revert y h₀ h₁\n"
            f"  intro x hx\n"
            f"  rw [Real.sqrt_eq_iff_sq_eq hx]\n"
            f"  swap\n"
            f"  norm_num\n"
            f"  intro h\n"
            f"  nlinarith\n\n"
            f"=== RULES YOU MUST ALWAYS FOLLOW ===\n"
            f"1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.\n"
            f"2. Use only variables that you have defined!\n"
            f"3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict, but use the arguments directly as in moogle_semantic_search(query=...).\n"
            f"4. Never write or execute Python code for proof steps or Lean code generation. Only use code_generator for Lean code.\n"
            f"5. When using code_generator, you must clearly state the Lean theorem statement and call the agent as in the correct example above.\n"
            f"6. After developing or updating a proof strategy, ALWAYS call the code_generator agent to generate and verify Lean code, even if the previous attempt failed. Every iteration must include a call to code_generator with the current strategy and lemmas.\n"
            f"7. Do not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable.\n"
            f"8. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.\n"
            f"9. Don't name any new variable with the same name as a tool.\n"
            f"10. Never create any notional variables in your code, as having these in your logs might derail you from the true variables.\n"
            f"11. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}\n"
            f"12. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.\n"
            f"13. Don't give up! You're in charge of solving the task, not providing directions to solve it.\n"
            f"14. CRITICAL: All final theorem proofs MUST use Lean 4 syntax with 'by' keyword. Use 'begin...end' only for complex multi-step proofs.\n"
            f"15. CRITICAL: Always include 'import MiniF2F.Minif2fImport' at the top of your Lean code.\n"
            f"16. CRITICAL: Use correct Lean 4 syntax: 'Finset.range' (capital F), not 'finset.range'.\n"
            f"17. CRITICAL: Keep proofs simple and direct. Avoid complex manual proofs that may timeout.\n"
            f"18. CRITICAL: Use ONLY these proven Lean 4 tactics: norm_num, linarith, nlinarith, ring, simp, rw, exact, apply, cases, induction, tauto, field_simp, ring_nf, assumption, contradiction, exfalso, by_contra, existsi, use, refine, constructor, split, left, right, intro, intros, revert, generalize, specialize, have, let, calc, suffices, by_cases, by_contradiction, push_neg, pull_out, push_in, clear, rename, change, unfold, delta, dsimp, rw_r, erw, rwa, rw_rule, rw_search, simp_rw, simp_all, simp_intros, simp_arith, norm_cast, push_cast, pull_cast, ring_exp, ring_nf, linarith!, nlinarith!, norm_num!, ring!, simp!, rw!, exact!, apply!, cases!, induction!, tauto!, field_simp!, ring_nf!, assumption!, contradiction!, exfalso!, by_contra!, existsi!, use!, refine!, constructor!, split!, left!, right!, intro!, intros!, revert!, generalize!, specialize!, have!, let!, calc!, suffices!, by_cases!, by_contradiction!, push_neg!, pull_out!, push_in!, clear!, rename!, change!, unfold!, delta!, dsimp!, rw_r!, erw!, rwa!, rw_rule!, rw_search!, simp_rw!, simp_all!, simp_intros!, simp_arith!, norm_cast!, push_cast!, pull_cast!, ring_exp!, ring_nf!.\n"
            f"19. CRITICAL: NEVER use tactics that don't exist in Lean 4. If unsure, stick to: norm_num, linarith, ring, simp, rw, exact, apply.\n"
            f"20. CRITICAL: NEVER invent new tactics or use tactics you're not sure exist. Stick to the proven list above.\n"
            f"21. CRITICAL: Use correct Lean 4 syntax: 'Finset.prod' instead of '∏', 'Finset.sum' instead of '∑', 'Finset.range' (capital F), not 'finset.range'.\n"
            f"22. CRITICAL: If a proof seems too complex, use 'sorry' and try a simpler approach.\n\n"
            f"=== REQUIRED PROOF FORMAT ===\n"
            f"Your final answer must be formatted exactly like this:\n"
            f"import MiniF2F.Minif2fImport\n\n"
            f"theorem theorem_name :\n"
            f"  statement_here :=\n"
            f"by\n"
            f"  proof_steps_here\n\n"
            f"For simple proofs:\n"
            f"import MiniF2F.Minif2fImport\n\n"
            f"theorem mathd_algebra_10 :\n"
            f"  abs ((120 : ℝ)/100 * 30 - 130/100 * 20) = 10 :=\n"
            f"by norm_num\n\n"
            f"For complex proofs:\n"
            f"import MiniF2F.Minif2fImport\n\n"
            f"theorem complex_example :\n"
            f"  statement_here :=\n"
            f"begin\n"
            f"  proof_step_1\n"
            f"  proof_step_2\n"
            f"  proof_step_3\n"
            f"end\n\n"
            f"=== YOUR TASK ===\n"
            f"Prove this theorem:\n{theorem['statement']}\n\n"
            f"Begin by analyzing the theorem and planning your approach!"
        )

        # Count tokens for the prompt
        if enc is not None and token_counter is not None:
            prompt_tokens = len(enc.encode(prompt))
            token_counter['total'] += prompt_tokens
            logger.debug(f"Prompt tokens: {prompt_tokens}")

        result = agent.run(prompt)
        
        # Count tokens for the result
        if enc is not None and token_counter is not None and result is not None:
            if isinstance(result, str):
                result_tokens = len(enc.encode(result))
            elif isinstance(result, dict):
                result_tokens = len(enc.encode(json.dumps(result)))
            else:
                result_tokens = 0
            token_counter['total'] += result_tokens
            logger.debug(f"Result tokens: {result_tokens}")
        
        # Analyze the result
        if result is None:
            logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (no result)")
            logger.debug(f"Agent result:\n{result}")
            return False
            
        if isinstance(result, dict):
            if result.get('success') is True:
                # Log the Lean code solution if present
                if 'lean_code' in result:
                    logger.info(f"Lean code solution for {theorem['name']}:\n{result['lean_code']}")
                logger.info(f"✅ Agent successfully generated proof for {theorem['name']}")
                logger.debug(f"Agent result:\n{result}")
                return True
            else:
                logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (agent returned success: False or error)")
                logger.debug(f"Agent result:\n{result}")
                return False
                
        if isinstance(result, str) and 'theorem' in result:
            # Log the Lean code solution
            logger.info(f"Lean code candidate for {theorem['name']}:\n{result}")
            verification = verify_lean_proof(result)
            if isinstance(verification, dict) and verification.get('success') is True:
                logger.info(f"✅ Agent successfully generated proof for {theorem['name']} (after verification)")
                logger.debug(f"Agent result:\n{result}")
                return True
            else:
                logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (verification failed)")
                logger.debug(f"Verification result:\n{verification}")
                return False
        
        logger.warning(f"❌ Agent failed to generate valid proof for {theorem['name']} (no valid result)")
        logger.debug(f"Agent result:\n{result}")
        return False
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Error during agent execution for {theorem['name']}: {error_message}")
        
        # Check for error code 402 (insufficient credits) - signal to stop processing
        if "Error code: 402" in error_message:
            logger.critical("❌ CRITICAL: Insufficient credits to continue.")
            logger.critical("Visit https://openrouter.ai/settings/credits to add more credits.")
            return "INSUFFICIENT_CREDITS"
        
        return False

def process_theorem_task(theorem: dict, max_steps: int, planning_interval: int, model_id: str, enc: Optional[tiktoken.Encoding]) -> tuple[Union[bool, str], int, str]:
    """
    A wrapper function to process a single theorem in a separate thread.
    Creates its own agent to ensure thread safety.
    """
    # Each thread gets its own agent to avoid state conflicts.
    agent = create_math_prover_agent(max_steps=max_steps, planning_interval=planning_interval, model_id=model_id)
    
    # Each thread manages its own token count.
    thread_token_counter = {'total': 0}
    
    # The original function can be called without modification.
    success = prove_theorem_with_agent(
        agent,
        theorem,
        max_steps=max_steps,
        enc=enc,
        token_counter=thread_token_counter
    )
    
    # Return everything needed by the main thread.
    return success, thread_token_counter['total'], theorem['name']

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
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Number of theorems to process in parallel (default: {DEFAULT_CONCURRENCY}).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use for theorem proving (default: {DEFAULT_MODEL}). Available models: {', '.join(AVAILABLE_MODELS)}")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Name of checkpoint to resume from (without .json extension).")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                        help="Name for saving checkpoint (without .json extension).")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N processed tasks (default: 5).")
    parser.add_argument("--list_checkpoints", action="store_true",
                        help="List all available checkpoints and exit.")
    args = parser.parse_args()

    # Validate model selection
    if args.model not in AVAILABLE_MODELS:
        logger.error(f"Invalid model '{args.model}'. Available models: {', '.join(AVAILABLE_MODELS)}")
        return

    # --- tiktoken setup ---
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except Exception as e:
        logger.warning(f"tiktoken not available or failed to load: {e}")
        enc = None
    token_counter = {'total': 0}

    MICRO_SUBSET_SIZE = args.subset_size
    VALID_JSON_PATH = args.json_file
    MAX_STEPS = args.max_steps
    PLANNING_INTERVAL = args.planning_interval
    CHECKPOINT_INTERVAL = args.checkpoint_interval
    CONCURRENCY = args.concurrency
    SELECTED_MODEL = args.model
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Handle checkpoint listing
    if args.list_checkpoints:
        checkpoints = list_checkpoints()
        if checkpoints:
            logger.info("Available checkpoints:")
            for checkpoint in checkpoints:
                logger.info(f"  - {checkpoint}")
        else:
            logger.info("No checkpoints found.")
        return

    logger.info("Starting Agent-based MiniF2F Lean Prover Benchmark...")
    logger.info(f"Configuration: Subset Size={MICRO_SUBSET_SIZE}, JSON File='{VALID_JSON_PATH}', Model='{SELECTED_MODEL}', Log Level='{args.log_level}', Max Steps={MAX_STEPS}, Planning Interval={PLANNING_INTERVAL}, Concurrency={CONCURRENCY}")
    if args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
    if args.save_checkpoint:
        logger.info(f"Will save checkpoint as: {args.save_checkpoint}")
    logger.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL} tasks")
    
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

    # Load checkpoint if specified
    start_index = 0
    results = {}
    if args.checkpoint:
        checkpoint_data = load_checkpoint(args.checkpoint)
        if checkpoint_data:
            results = checkpoint_data['results']
            start_index = checkpoint_data['processed_count']
            logger.info(f"Resuming from task {start_index + 1}/{len(micro_subset)}")
        else:
            logger.warning(f"Failed to load checkpoint '{args.checkpoint}', starting from beginning.")
            start_index = 0

    # Agent creation is moved into the worker threads to ensure thread safety.

    logger.info("\n--- Running Agent-based Theorem Proving on Micro-Subset ---")
    insufficient_credits = False
    
    processed_count = start_index
    total_tasks_to_run = len(micro_subset) - start_index
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_theorem_name = {
            executor.submit(process_theorem_task, theorem, MAX_STEPS, PLANNING_INTERVAL, SELECTED_MODEL, enc): theorem['name']
            for theorem in micro_subset[start_index:]
        }
        
        for future in concurrent.futures.as_completed(future_to_theorem_name):
            theorem_name = future_to_theorem_name[future]
            try:
                success, tokens_used, _ = future.result()
                
                token_counter['total'] += tokens_used
                results[theorem_name] = success
                
                if success == "INSUFFICIENT_CREDITS":
                    insufficient_credits = True
                    logger.critical(f"Stopping processing due to insufficient credits. Processed {processed_count - start_index}/{total_tasks_to_run} tasks in this run.")
                    # Attempt to cancel remaining futures
                    for f in future_to_theorem_name:
                        f.cancel()
                    break
                
            except Exception as e:
                logger.error(f"❌ Error processing theorem {theorem_name}: {e}")
                results[theorem_name] = False
            
            processed_count += 1
            logger.info(f"Progress: {processed_count}/{len(micro_subset)} total tasks processed.")
            
            # Save checkpoint periodically
            if args.save_checkpoint and (processed_count - start_index) > 0 and (processed_count - start_index) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(results, processed_count, len(micro_subset), args.save_checkpoint, SELECTED_MODEL)

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

    # --- Token usage summary ---
    if enc is not None:
        logger.info(f"Total tokens used for prompts and results: {token_counter['total']}")
    else:
        logger.info("Token counting was not available (tiktoken not installed or failed to load).")
    
    # Save final checkpoint if requested
    if args.save_checkpoint:
        save_checkpoint(results, processed_count, len(micro_subset), args.save_checkpoint, SELECTED_MODEL)
    
    # --- Handle insufficient credits error ---
    if insufficient_credits:
        logger.critical("\n❌ PROGRAM TERMINATED: Insufficient credits to continue processing.")
        logger.critical("Add more credits at https://openrouter.ai/settings/credits and try again.")
        logger.critical(f"You can resume from checkpoint '{args.save_checkpoint}' when you have more credits.")
        sys.exit(1)

if __name__ == "__main__":
    main()