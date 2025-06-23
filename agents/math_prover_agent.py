import sys
from pathlib import Path

# Add the parent directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import CodeAgent, OpenAIServerModel
from config import DEFAULT_MODEL, OPENROUTER_API_BASE, OPENROUTER_API_KEY, validate_config
from agents.tools import verify_lean_proof

# Validate configuration before proceeding
try:
    validate_config()
except (ValueError, FileNotFoundError) as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

model = OpenAIServerModel(
    model_id=DEFAULT_MODEL,
    api_base=OPENROUTER_API_BASE,
    api_key=OPENROUTER_API_KEY,
)

agent = CodeAgent(tools=[verify_lean_proof], model=model)

theorem = r"""
theorem mathd_algebra_182
  (y : ℂ) :
  7 * (3 * y + 2) = 21 * y + 14 :=
begin
  sorry
end
"""

result = agent.run(
    f"You are a Lean 3.42.1 theorem prover. Your task is to generate Lean code that proves the given theorem.\n\n"
    f"CRITICAL: You must ONLY generate Lean code and use the verify_lean_proof tool. DO NOT write any Python code, mathematical analysis, or calculations.\n\n"
    f"WORKFLOW:\n"
    f"1. Look at the theorem statement\n"
    f"2. Generate a complete Lean theorem with proper tactics (replace 'sorry' with Lean tactics like norm_num, simp, rw, exact, etc.)\n"
    f"3. Use verify_lean_proof(theorem_statement) to check your Lean code\n"
    f"4. If verification fails, fix the Lean code and try again\n"
    f"5. Return ONLY the final verified Lean theorem\n\n"
    f"LEAN TACTICS YOU CAN USE:\n"
    f"- norm_num: for numerical calculations\n"
    f"- simp: for simplification\n"
    f"- rw: for rewriting with hypotheses\n"
    f"- exact: for exact proofs\n"
    f"- apply: for applying lemmas\n"
    f"- cases: for case analysis\n"
    f"- induction: for induction\n"
    f"- have: for introducing new hypotheses\n"
    f"- proof blocks: use {{ }} for subproofs\n\n"
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
    f"Prove this theorem:\n{theorem}"
)
print(result)