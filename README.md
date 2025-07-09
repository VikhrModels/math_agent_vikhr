# Math Agent Vikhr - Automated Lean Theorem Prover

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project uses Large Language Models (LLMs) to automatically generate proofs for mathematical theorems written in the Lean 3 programming language. It is designed to work with the `miniF2F` benchmark, a collection of formal-to-formal theorems from high-school math competitions.

## 📄 License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details. This license aligns with the miniF2F/lean folder which is also released under the Apache License to maintain compatibility with Lean's mathlib license.

## 🏗️ Project Structure

The project has been reorganized with a modular, maintainable architecture:

```
math_agent_vikhr/
├── 📄 config.py                   # Centralized configuration management
├── 📁 agents/                     # AI agents package
│   ├── 📄 __init__.py             # Package initialization
│   ├── 📄 math_prover_agent.py    # Main theorem proving agent
│   └── 📄 tools.py                # Custom tools and utilities
├── 📄 process_lean.py             # Lean file parser and processor
├── 📄 benchmark_sonnet_only.py    # Benchmark script for testing sonnet
├── 📄 README.md                   # This documentation
├── 📄 requirements.txt            # Python dependencies
├── 📄 valid.json                  # Processed theorems dataset
├── 📁 miniF2F/                    # Git submodule with mathematical problems
├── 📁 log/                        # Execution logs
├── 📁 tmp/                        # Temporary files
└── 📁 venv/                       # Virtual environment
```

## 🤖 AI Agents System

### Overview

The `agents/` package implements a sophisticated multi-agent system for automated theorem proving using the `smolagents` framework. The system consists of two main agents working together:

1. **Idea Generator Agent**: Analyzes theorem statements, searches for relevant lemmas, and develops proof strategies
2. **Code Generator Agent**: Generates and verifies Lean code based on the strategies provided

### Agent Architecture

#### `agents/math_prover_agent.py`

This is the main entry point for the agent-based theorem proving system.

**Key Functions:**

- `create_math_prover_agent()`: Creates a multi-agent system with idea and code generation capabilities
- `prove_theorem_with_agent()`: Executes a single theorem proof using the agent system
- `main()`: Orchestrates the complete benchmark process

**Agent Configuration:**
```python
from agents import create_math_prover_agent

# Create agent with custom parameters
agent = create_math_prover_agent(
    max_steps=10,           # Maximum agent steps per theorem
    planning_interval=1      # Planning frequency
)
```

**Multi-Agent System:**
- **Idea Generator**: Uses `moogle_semantic_search` tool to find relevant lemmas
- **Code Generator**: Uses `verify_lean_proof` tool to generate and verify Lean code
- **Managed Agents**: Code generator is managed by the idea generator

#### `agents/tools.py`

Contains the custom tools that agents use for theorem proving:

**1. VerifyLeanProof Tool**
```python
class VerifyLeanProof(Tool):
    """
    Verifies Lean 3.42.1 mathematical proofs by compiling them within the miniF2F project environment.
    
    Features:
    - Creates temporary files within miniF2F project structure
    - Handles both ':= sorry' and ':= begin sorry end' formats
    - Validates compilation success and produces .olean files
    - Comprehensive error handling and timeout management
    - Cleans up temporary files automatically
    """
```

**Usage:**
```python
from agents.tools import verify_lean_proof

result = verify_lean_proof("""
theorem example : 2 + 2 = 4 := 
begin
  norm_num
end
""")

# Returns: {'success': True, 'output': 'compilation output'}
```

**2. MoogleSemanticSearch Tool**
```python
class MoogleSemanticSearch(Tool):
    """
    Performs semantic search for theorems, lemmas, and mathematical structures via moogle.ai.
    
    Features:
    - Searches Lean mathlib for relevant mathematical declarations
    - Returns structured data with declaration names, code, and documentation
    - Handles brotli compression and JSON parsing
    - Comprehensive error handling for network requests
    """
```

**Usage:**
```python
from agents.tools import moogle_semantic_search

results = moogle_semantic_search("real number arithmetic lemma")
# Returns structured data about relevant lemmas and theorems
```

#### `agents/__init__.py`

Package initialization with version information and exports:
```python
__version__ = "1.0.0"
__author__ = "Math Agent Vikhr Team"
```

## 🚀 Usage

### Quick Start

1. **Setup Environment**:
   ```bash
   git clone --recurse-submodules https://github.com/umbra2728/math_agent_vikhr.git
   cd math_agent_vikhr
   pip install -r requirements.txt
   export OPENROUTER_API_KEY="your_key_here"
   ```

2. **Process Lean Files**:
   ```bash
   python process_lean.py
   ```

3. **Run Agent-based Benchmark**:
   ```bash
   python agents/math_prover_agent.py
   ```

### Agent-based Theorem Proving

The agent system provides sophisticated theorem proving capabilities:

#### Basic Usage
```bash
# Run with default settings
python agents/math_prover_agent.py

# Custom subset size
python agents/math_prover_agent.py --subset_size 20

# Debug logging
python agents/math_prover_agent.py --log_level DEBUG

# Custom model
python agents/math_prover_agent.py --model "anthropic/claude-sonnet-4"
```

#### Advanced Configuration

**Agent Parameters:**
- `--max_steps`: Maximum agent steps per theorem (default: 10)
- `--planning_interval`: Planning frequency (default: 1)
- `--subset_size`: Number of theorems to test (default: 10)
- `--json_file`: Path to theorems JSON file
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Checkpoint System:**
```bash
# Save checkpoint every 5 tasks
python agents/math_prover_agent.py --save_checkpoint my_run --checkpoint_interval 5

# Resume from checkpoint
python agents/math_prover_agent.py --checkpoint my_run

# List available checkpoints
python agents/math_prover_agent.py --list_checkpoints
```

#### Understanding Agent Behavior

**Agent Workflow:**
1. **Analysis**: Agent analyzes the theorem statement
2. **Search**: Uses semantic search to find relevant lemmas
3. **Strategy**: Develops a proof strategy based on found lemmas
4. **Generation**: Generates Lean code using the strategy
5. **Verification**: Compiles and verifies the generated proof
6. **Iteration**: Refines the proof if verification fails

**Agent Limits:**
- **max_steps**: Controls how many times the agent can "think" and generate new approaches
- **planning_interval**: Determines how frequently the agent plans its next action
- **timeout**: Prevents the agent from running indefinitely on a single theorem

### Traditional LLM Benchmark

For comparison, you can also run the traditional benchmark:

```bash
python benchmark_sonnet_only.py
```

## 🔧 Configuration

### `config.py` - Centralized Configuration

All project settings are centralized in `config.py`:

- **API Configuration**: OpenRouter API settings
- **Model Configuration**: Available LLM models and defaults
- **Lean Configuration**: File paths and timeouts
- **Agent Configuration**: Default parameters and settings
- **Logging Configuration**: Log formats and file paths

### Environment Variables

Set these environment variables before running:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export MATH_AGENT_MODEL="anthropic/claude-sonnet-4"  # Optional: override default model
export MATH_AGENT_SUBSET_SIZE="10"                   # Optional: override subset size
```

## 🛠️ Development

### Adding New Tools

To add custom tools to the agents:

1. **Define tools in `agents/tools.py`:**
   ```python
   from smolagents import Tool
   
   class MyCustomTool(Tool):
       name = "my_custom_tool"
       description = "Description of what this tool does"
       
       inputs = {
           "param": {
               "type": "string",
               "description": "Description of the parameter"
           }
       }
       
       output_type = "object"
       
       def forward(self, param: str) -> dict:
           # Tool implementation
           return {"result": "success"}
   ```

2. **Create tool instance:**
   ```python
   my_custom_tool = MyCustomTool()
   ```

3. **Use in agents:**
   ```python
   from agents.tools import my_custom_tool
   
   agent = create_math_prover_agent(tools=[my_custom_tool])
   ```

### Extending Agent Capabilities

**Adding New Agent Types:**
```python
def create_math_prover_agent(agent_type="idea", **kwargs):
    if agent_type == "idea":
        return CodeAgent(
            tools=[moogle_semantic_search],
            managed_agents=[code_agent],
            **kwargs
        )
    elif agent_type == "code":
        return CodeAgent(
            tools=[verify_lean_proof],
            **kwargs
        )
```

**Custom Agent Prompts:**
```python
def create_custom_prompt(theorem: dict) -> str:
    return f"""
    You are an expert Lean theorem prover.
    
    Theorem: {theorem['statement']}
    
    Your task is to:
    1. Analyze the theorem
    2. Search for relevant lemmas
    3. Generate a valid Lean proof
    4. Verify the proof compiles successfully
    """
```

### Configuration Management

- Add new settings to `config.py`
- Use environment variables for sensitive data
- Implement validation in `validate_config()`
- Add environment-specific overrides as needed

## 📊 Results and Monitoring

### Logging System

The agent system provides comprehensive logging:

- **Agent Logs**: `log/agent_benchmark.log`
- **Tool Logs**: Individual tool execution logs
- **Checkpoint Data**: Progress saved in `tmp/checkpoints/`
- **Token Usage**: Tracks LLM token consumption

### Checkpoint System

**Automatic Checkpoints:**
```bash
# Save every 5 tasks
python agents/math_prover_agent.py --save_checkpoint my_run --checkpoint_interval 5
```

**Manual Checkpoints:**
```python
from agents.math_prover_agent import save_checkpoint, load_checkpoint

# Save progress
save_checkpoint(results, processed_count, total_count, "my_checkpoint")

# Load progress
checkpoint_data = load_checkpoint("my_checkpoint")
```

### Performance Metrics

The system tracks:
- **Success Rate**: Percentage of successfully proven theorems
- **Token Usage**: Total tokens consumed by LLM interactions
- **Processing Time**: Time per theorem and total benchmark time
- **Error Analysis**: Detailed error logs for failed proofs

## 🔒 Security and Best Practices

### API Key Management
- Store API keys in environment variables
- Never commit sensitive data to version control
- Use different keys for development and production

### Error Handling
- Comprehensive timeout management
- Graceful handling of API failures
- Automatic cleanup of temporary files
- Detailed error logging for debugging

### Resource Management
- Configurable timeouts for Lean compilation
- Memory-efficient processing of large theorem sets
- Automatic cleanup of temporary files and processes

## 🤝 Contributing

### Development Guidelines

1. **Follow the modular structure**
   - Keep agent logic in `agents/` package
   - Use centralized configuration in `config.py`
   - Add proper logging and error handling

2. **Agent Development**
   - Test new tools thoroughly
   - Document tool interfaces clearly
   - Maintain backward compatibility

3. **Configuration**
   - Add new settings to `config.py`
   - Use environment variables for sensitive data
   - Implement proper validation

4. **Testing**
   - Test with small theorem subsets first
   - Verify Lean compilation works correctly
   - Check token usage and costs

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters
- Add comprehensive docstrings
- Include error handling for all external calls

## 📄 License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### License Compatibility

This project uses the Apache License 2.0 to maintain compatibility with:
- **miniF2F/lean**: The mathematical problems and Lean formalizations are also under Apache License 2.0
- **Lean's mathlib**: The Lean mathematical library uses Apache License 2.0
- **Open source ecosystem**: Apache License 2.0 is widely compatible with other open source licenses

### Attribution

This project uses the **MiniF2F** benchmark dataset for mathematical theorem proving evaluation. MiniF2F is a formal mathematics benchmark consisting of exercise statements from olympiads (AMC, AIME, IMO) and high-school mathematics.

**Citation for MiniF2F:**
```
@article{zheng2021minif2f,
  title={MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics},
  author={Zheng, Kunhao and Han, Jesse Michael and Polu, Stanislas},
  journal={arXiv preprint arXiv:2109.00110},
  year={2021}
}
```

**MiniF2F Repository:** https://github.com/openai/miniF2F

The Lean formalizations used in this project are from the `miniF2F/lean` folder, which is released under the Apache License 2.0 to align with Lean's mathlib license.

### Copyright Notice

To apply the Apache License to your work, attach the following boilerplate notice, with the fields enclosed by brackets "[]" replaced with your own identifying information:

```
Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Features

- **Multi-Agent System**: Sophisticated agent-based theorem proving using smolagents
- **Lean 3.42.1 Integration**: Full integration with Lean compiler and mathlib
- **Semantic Search**: Moogle.ai integration for finding relevant lemmas and theorems
- **Proof Verification**: Automatic compilation and verification of generated proofs
- **Checkpoint System**: Resume long-running benchmarks from saved progress
- **Token Tracking**: Monitor LLM token usage and costs
- **Stratified Sampling**: Representative testing with preserved solved/unsolved ratios
- **Comprehensive Logging**: Detailed logs for debugging and analysis
- **Error Handling**: Robust error handling for network, compilation, and API issues
- **Configuration Management**: Centralized configuration with environment overrides