import sys
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
import tiktoken
from typing import Union, Optional
import concurrent.futures

# Add the parent directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import CodeAgent, OpenAIServerModel
from config import (
    DEFAULT_MODEL, AVAILABLE_MODELS,
    OPENROUTER_API_BASE, OPENROUTER_API_KEY,
    MINIF2F_DIR, LOG_DIR, TMP_DIR, LEAN_OUTPUT_FILE,
    DEFAULT_SUBSET_SIZE, DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_STEPS, DEFAULT_PLANNING_INTERVAL,
    DEFAULT_CONCURRENCY,
    validate_config, validate_provider_credentials,
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

def _ensure_run_dir(run_id: str) -> Path:
    run_dir = CHECKPOINT_DIR / f"run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_stage_checkpoint(
    run_id: str,
    stage_index: int,
    stages_total: int,
    stage_results: dict,
    cumulative_results: dict,
    processed_count: int,
    total_count: int,
    model_id: str,
    provider: str,
    tokens_used_total: int | None = None,
) -> None:
    """Save a per-stage checkpoint inside a run-specific directory.

    Always called after each processed task and at the end of a stage.
    """
    run_dir = _ensure_run_dir(run_id)
    now_iso = datetime.now().isoformat(timespec="seconds")
    unsolved_remaining = [name for name, ok in cumulative_results.items() if ok is not True]
    checkpoint_data = {
        "run_id": run_id,
        "timestamp": now_iso,
        "stage_index": stage_index,
        "stages_total": stages_total,
        "model": model_id,
        "provider": provider,
        "processed_count": processed_count,
        "total_count": total_count,
        "solved_stage": sum(1 for v in stage_results.values() if v is True),
        "solved_cumulative": sum(1 for v in cumulative_results.values() if v is True),
        "pass_rate_stage": (sum(1 for v in stage_results.values() if v is True) / total_count * 100) if total_count > 0 else 0.0,
        "unsolved_remaining": unsolved_remaining,
        "results_stage": stage_results,
        "results_cumulative": cumulative_results,
    }
    if tokens_used_total is not None:
        checkpoint_data["tokens_used_total"] = tokens_used_total
    stage_file = run_dir / f"stage-{stage_index}.json"
    try:
        with stage_file.open('w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Stage checkpoint saved: {stage_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save stage checkpoint: {e}")

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

def create_math_prover_agent(max_steps: int = DEFAULT_MAX_STEPS,
                             planning_interval: int = DEFAULT_PLANNING_INTERVAL,
                             model_id: str = DEFAULT_MODEL):
    """
    Re-architected multi-agent Lean prover based on a top-down planner.

    Returns the planner/orchestrator agent that internally manages:
        • search_agent — semantic search specialist powered by `moogle_semantic_search`
        • code_agent   — Lean code generator/verifier using `verify_lean_proof`
    """
    # Validate configuration & credentials
    try:
        validate_config()
    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)

    validate_provider_credentials("openrouter")
    api_base = OPENROUTER_API_BASE
    api_key = OPENROUTER_API_KEY

    # Shared model across sub-agents (cheaper & coherent)
    model = OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
    )

    # --- Search Agent ---------------------------------------------------
    search_agent = CodeAgent(
        tools=[moogle_semantic_search],
        model=model,
        max_steps=6,
        planning_interval=1,
        name="search_agent",
        description=(
            "You are a semantic search specialist for Lean proofs. "
            "Given a goal and its context you issue PARALLEL queries to the `moogle_semantic_search` tool in a single "
            "Python code block, aggregate and return the most useful mathlib facts together with short hints on how "
            "they might be applied (rw/apply/simp/etc.). "
            "Batch all queries; never run them one by one in distinct steps."
        ),
    )

    # --- Code Agent -----------------------------------------------------
    code_agent = CodeAgent(
        tools=[verify_lean_proof],
        model=model,
        max_steps=10,
        planning_interval=1,
        name="code_agent",
        description=(
            "Generates Lean-4 code skeletons and iteratively replaces `sorry` with real proofs. "
            "In a single iteration it may prepare SEVERAL alternative proofs for the current lemma and verify them via "
            "`verify_lean_proof`, choosing the best compiling version. "
            "NEVER write Python for mathematical reasoning — only Lean. Always include `import MiniF2F.Minif2fImport` "
            "at the top and ensure the final file contains NO `sorry`."
        ),
    )

    # --- Planner / Orchestrator ----------------------------------------
    planner_agent = CodeAgent(
        tools=[verify_lean_proof, moogle_semantic_search],  # quick fast-path compilations
        managed_agents=[search_agent, code_agent],
        model=model,
        max_steps=max_steps,
        planning_interval=planning_interval,
        name="planner_agent",
        description=(
            "Top-down planner that decomposes a theorem into lemma stubs, assigns difficulty + criticality, "
            "and orchestrates proof construction. It decides when to call `search_agent` for facts or `code_agent` "
            "for Lean code, manages token budgets, and revises the plan when Lean feedback suggests better "
            "formulations. Fast-path easy goals by directly generating small Lean snippets and compiling them via "
            "`verify_lean_proof`."
        ),
    )

    return planner_agent

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
        # --- Rewritten planner-centric prompt ---
        prompt = f"""SYSTEM OVERVIEW:
You control a TEAM of specialised agents that solve Lean theorems via a top-down → bottom-up pipeline:
  • planner_agent  (you) — decomposes the goal into lemma stubs, assigns difficulty/criticality and orchestrates.
  • search_agent   — batches semantic searches with moogle_semantic_search and returns curated facts + hints.
  • code_agent     — writes Lean-4 code: first a skeleton with `sorry`, then iteratively replaces `sorry` with proofs,
                     trying SEVERAL variants per iteration and keeping the first compiling proof.
  • verify_lean_proof — Lean compiler wrapper, the single source of truth.

IMPORTANT: FOCUS ON CODE GENERATION FIRST, not search. Only use search_agent when you have a specific fact you need.
Most theorems can be solved with standard Lean tactics and mathematical reasoning.

WORKFLOW (high-level):
  1. NORMALISE the theorem (simplify quantifiers, bring into canonical forms).
  2. BUILD a DAG of lemma stubs (no proofs) with <complexity, criticality>. Limit recursion depth to 3.
  3. Generate a Lean skeleton (imports + main theorem + lemma stubs). Ensure it compiles.
  4. LOOP while there are `sorry`:
     a. pick the highest-priority lemma (critical × moderate complexity).
     b. attempt cheap tactics first (simp/rw/norm_num etc). If solved ⇒ substitute and recompile.
     c. if needed, call search_agent ONCE with a batch of re-phrased queries. Integrate returned facts.
     d. ask code_agent to produce ≤3 candidate proofs using gathered facts. Accept the first compiling one.
     e. if Lean feedback shows type mismatch / missing instances / timeout ⇒ revise stub or split it.
  5. When no `sorry` remain, return the full Lean code as FINAL_ANSWER.

BUDGETS: planner iterations ≤ {max_steps}; code_agent iterations per lemma ≤ 7; search_agent refinements ≤ 3.

FAST-PATH: if the current goal is recognised as "easy" (single tactic likely) you may bypass sub-agents: craft a
small Lean snippet yourself and verify via verify_lean_proof directly.

SEMAPHORES: Always batch queries to search_agent, never sequential single queries. Do not request Python maths.
All Lean code MUST import MiniF2F.Minif2fImport and contain no `sorry` at termination.

FEW-SHOT EXAMPLES:

EXAMPLE 1 (Medium complexity - algebraic manipulation):
Theorem: mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4) (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4
Solution approach: Substitute h₀ into h₁, solve for k using algebra and norm_num.

EXAMPLE 2 (Hard complexity - number theory with induction):
Theorem: induction_sum2kp1npqsqm1 (n : ℕ) : ∑ k ∈ Finset.range n, (2 * k + 3) = (n + 1) ^ 2 - 1
Solution approach: Use mathematical induction, base case n=0, inductive step with sum manipulation.

EXAMPLE 3 (Hard complexity - complex analysis):
Theorem: amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) : ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36
Solution approach: Use properties of complex numbers, geometric series, and power reduction.

EXAMPLE 4 (Medium complexity - function analysis):
Theorem: amc12_2000_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) : f 600 = 5 / 2
Solution approach: Use functional equation to find f(x) = c/x, solve for constant c.

TASK — prove the following theorem:

{theorem['statement']}

First: classify overall difficulty (easy / medium / hard) and outline the DAG plan (list lemma names + one-line goal).
Focus on generating Lean code directly rather than searching for facts."""

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

def process_theorem_task(theorem: dict,
                         max_steps: int,
                         planning_interval: int,
                         model_id: str,
                         enc: Optional[tiktoken.Encoding]) -> tuple[Union[bool, str], int, str]:
    """
    A wrapper function to process a single theorem in a separate thread.
    Creates its own agent to ensure thread safety.
    """
    # Each thread gets its own agent to avoid state conflicts.
    agent = create_math_prover_agent(max_steps=max_steps,
                                     planning_interval=planning_interval,
                                     model_id=model_id)
    
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
                        help=f"Model to use (default: {DEFAULT_MODEL}). Available: {', '.join(AVAILABLE_MODELS)}")
    # Provider is fixed to OpenRouter; CLI option removed
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Name of checkpoint to resume from (without .json extension).")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                        help="Name for saving checkpoint (without .json extension).")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N processed tasks (default: 5).")
    parser.add_argument("--list_checkpoints", action="store_true",
                        help="List all available checkpoints and exit.")
    # Stages (multi-pass)
    parser.add_argument("--stages", type=int, default=1,
                        help="Number of passes over the dataset. 1 = single pass; 2 = rerun on unsolved, etc.")
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
    SELECTED_PROVIDER = "openrouter"
    
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

    # Prepare staged processing
    total_tasks_initial = len(micro_subset)
    name_to_theorem = {t['name']: t for t in micro_subset}
    stages_total = max(1, int(getattr(args, 'stages', 1)))
    logger.info(f"Stages requested: {stages_total}")

    # Resume support (legacy single-file checkpoint)
    cumulative_results: dict[str, Union[bool, str]] = {}
    if args.checkpoint:
        checkpoint_data = load_checkpoint(args.checkpoint)
        if checkpoint_data:
            cumulative_results = checkpoint_data.get('results', {})
            logger.info(f"Loaded previous results from checkpoint '{args.checkpoint}'.")
        else:
            logger.warning(f"Failed to load checkpoint '{args.checkpoint}', starting fresh.")

    # Run identifier (timestamp) for this multi-stage run
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    insufficient_credits = False

    # Stage loop
    current_tasks = micro_subset
    for stage_index in range(1, stages_total + 1):
        if stage_index == 1:
            tasks_this_stage = current_tasks
        else:
            # Only unsolved from cumulative_results
            tasks_this_stage = [name_to_theorem[name] for name in name_to_theorem.keys() if cumulative_results.get(name) is not True]
        if not tasks_this_stage:
            logger.info(f"Stage {stage_index}/{stages_total}: nothing to process (all solved).")
            # Save an empty stage checkpoint to document progress
            save_stage_checkpoint(
                run_id=run_id,
                stage_index=stage_index,
                stages_total=stages_total,
                stage_results={},
                cumulative_results=cumulative_results,
                processed_count=0,
                total_count=0,
                model_id=SELECTED_MODEL,
                provider=SELECTED_PROVIDER,
                tokens_used_total=token_counter['total'] if enc is not None else None,
            )
            continue

        logger.info(f"\n--- Stage {stage_index}/{stages_total}: Processing {len(tasks_this_stage)} tasks ---")
        stage_results: dict[str, Union[bool, str]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_theorem_name = {
                executor.submit(process_theorem_task, theorem, MAX_STEPS, PLANNING_INTERVAL, SELECTED_MODEL, enc): theorem['name']
                for theorem in tasks_this_stage
            }

            processed_in_stage = 0
            for future in concurrent.futures.as_completed(future_to_theorem_name):
                theorem_name = future_to_theorem_name[future]
                try:
                    success, tokens_used, _ = future.result()
                    token_counter['total'] += tokens_used
                    stage_results[theorem_name] = success
                    cumulative_results[theorem_name] = success
                    if success == "INSUFFICIENT_CREDITS":
                        insufficient_credits = True
                        logger.critical("Stopping processing due to insufficient credits during stage.")
                        # Attempt to cancel remaining futures
                        for f in future_to_theorem_name:
                            f.cancel()
                        break
                except Exception as e:
                    logger.error(f"❌ Error processing theorem {theorem_name}: {e}")
                    stage_results[theorem_name] = False
                    cumulative_results[theorem_name] = False

                processed_in_stage += 1
                logger.info(f"Stage {stage_index}: progress {processed_in_stage}/{len(tasks_this_stage)}")

                # Always save a stage checkpoint after each processed task
                save_stage_checkpoint(
                    run_id=run_id,
                    stage_index=stage_index,
                    stages_total=stages_total,
                    stage_results=stage_results,
                    cumulative_results=cumulative_results,
                    processed_count=processed_in_stage,
                    total_count=len(tasks_this_stage),
                    model_id=SELECTED_MODEL,
                    provider=SELECTED_PROVIDER,
                    tokens_used_total=token_counter['total'] if enc is not None else None,
                )

            # End-of-stage checkpoint (ensures a full snapshot regardless of where the loop ended)
            save_stage_checkpoint(
                run_id=run_id,
                stage_index=stage_index,
                stages_total=stages_total,
                stage_results=stage_results,
                cumulative_results=cumulative_results,
                processed_count=len(stage_results),
                total_count=len(tasks_this_stage),
                model_id=SELECTED_MODEL,
                provider=SELECTED_PROVIDER,
                tokens_used_total=token_counter['total'] if enc is not None else None,
            )

        if insufficient_credits:
            break

    # Final reporting
    logger.info("\n--- Summary of Results (Cumulative) ---")
    solved_count = sum(1 for success in cumulative_results.values() if success is True)
    total_count = len(name_to_theorem)
    for name in sorted(name_to_theorem.keys()):
        success = cumulative_results.get(name, False)
        status = "PASSED" if success is True else ("ERROR" if isinstance(success, str) else "FAILED")
        logger.info(f"{name}: {status}")

    logger.info(f"\nTotal tasks processed: {len(cumulative_results)}")
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

    # Keep legacy single-file checkpoint support if requested
    if args.save_checkpoint:
        save_checkpoint(cumulative_results, len(cumulative_results), len(name_to_theorem), args.save_checkpoint, SELECTED_MODEL)

    # --- Handle insufficient credits error ---
    if insufficient_credits:
        logger.critical("\n❌ PROGRAM TERMINATED: Insufficient credits to continue processing.")
        logger.critical("Add more credits at https://openrouter.ai/settings/credits and try again.")
        logger.critical("You can resume the next run; per-stage checkpoints are preserved.")
        sys.exit(1)

if __name__ == "__main__":
    main()