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

result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)