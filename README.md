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

## 🔧 Configuration

### `config.py` - Centralized Configuration

All project settings are centralized in `config.py`:

- **API Configuration**: OpenRouter API settings
- **Model Configuration**: Available LLM models and defaults
- **Lean Configuration**: File paths and timeouts
- **Agent Configuration**: Default parameters and settings
- **Logging Configuration**: Log formats and file paths

Key features:
- Environment-specific overrides
- Configuration validation
- Centralized path management
- Security through environment variables

### Environment Variables

Set these environment variables before running:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export MATH_AGENT_MODEL="anthropic/claude-sonnet-4"  # Optional: override default model
export MATH_AGENT_SUBSET_SIZE="10"                   # Optional: override subset size
```

## 🤖 AI Agents

### `agents/` Package

The `agents/` package contains all AI agent logic:

#### `agents/math_prover_agent.py`
- **MathProverAgent**: Main agent class for theorem proving
- Supports both CodeAgent and ToolCallingAgent types
- Integrated with centralized configuration
- Comprehensive logging and error handling

#### `agents/tools.py`
- Custom tools and utilities for agents
- Lean verification utilities
- Common helper functions
- Foundation for future tool implementations

#### `agents/__init__.py`
- Package initialization
- Convenient imports: `from agents import create_math_prover_agent`

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

3. **Run Benchmark**:
   ```bash
   python benchmark_sonnet_only.py
   ```

### Using the MathProverAgent

```python
from agents import create_math_prover_agent

# Create a CodeAgent
agent = create_math_prover_agent(agent_type="code")

# Run a task
response = agent.run("What is the 10th prime number? Use code to find it.")

# Create a ToolCallingAgent
tool_agent = create_math_prover_agent(agent_type="tool_calling")
```

### Command Line Options

The benchmark script supports various options:

```bash
# Basic usage
python benchmark_sonnet_only.py

# Custom subset size
python benchmark_sonnet_only.py --subset_size 20

# Different model
python benchmark_sonnet_only.py --model "google/gemini-pro"

# Custom JSON file
python benchmark_sonnet_only.py --json_file /path/to/theorems.json

# Debug logging
python benchmark_sonnet_only.py --log_level DEBUG
```

## 🔄 Workflow

The core workflow remains the same but is now more modular:

1. **Configuration**: `config.py` validates all settings
2. **Parsing**: `process_lean.py` extracts theorems from Lean files
3. **Agent Creation**: `agents/math_prover_agent.py` creates configured agents
4. **Proof Generation**: LLM generates proofs using the agent
5. **Verification**: Lean compiler validates the proofs
6. **Reporting**: Results are logged and summarized

## 🛠️ Development

### Adding New Tools

To add custom tools to the agents:

1. Define tools in `agents/tools.py`:
   ```python
   from smolagents import tool
   
   @tool
   def my_custom_tool(param: str) -> str:
       """Description of what this tool does.
       
       Args:
           param: Description of the parameter
       """
       # Tool implementation
       return result
   ```

2. Use in agents:
   ```python
   from agents.tools import my_custom_tool
   
   agent = create_math_prover_agent(tools=[my_custom_tool])
   ```

### Configuration Management

- Add new settings to `config.py`
- Use environment variables for sensitive data
- Implement validation in `validate_config()`
- Add environment-specific overrides as needed

## 📊 Results

The system provides detailed logging and reporting:

- **Logs**: Stored in `log/llm_requests.log`
- **Micro-subset**: Selected tasks saved to `micro_subset.txt`
- **Console Output**: Real-time progress and results
- **Pass Rate**: Percentage of successfully proven theorems

## 🔒 Security

- API keys stored in environment variables
- No hardcoded sensitive information
- Configuration validation prevents insecure settings
- Lean verification runs in controlled environment

## 🤝 Contributing

1. Follow the modular structure
2. Use centralized configuration
3. Add proper logging
4. Include error handling

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

- Agent-based theorem proving using smolagents
- Lean 4 proof verification
- Support for multiple LLM models via OpenRouter
- Stratified sampling for representative testing
- Comprehensive logging and error handling

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

3. Initialize the miniF2F submodule:
```bash
git submodule update --init --recursive
```

## Usage

### Agent-based Theorem Proving

Run the agent-based benchmark:

```bash
python agents/math_prover_agent.py
```

#### Agent Configuration Options

You can control the agent's behavior with these parameters:

- `--max_iterations`: Maximum number of agent iterations per theorem (default: 5)
- `--max_tool_calls`: Maximum number of tool calls per theorem (default: 10)  
- `--agent_timeout`: Timeout in seconds for agent execution per theorem (default: 300)
- `--subset_size`: Number of theorems to test (default: 10)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Example with custom limits:
```bash
python agents/math_prover_agent.py \
  --max_iterations 3 \
  --max_tool_calls 5 \
  --agent_timeout 120 \
  --subset_size 5 \
  --log_level DEBUG
```

#### Understanding Agent Limits

- **max_iterations**: Controls how many times the agent can "think" and generate new approaches
- **max_tool_calls**: Limits the number of times the agent can call the `verify_lean_proof` tool
- **agent_timeout**: Prevents the agent from running indefinitely on a single theorem

Lower limits make the agent faster but may reduce success rate. Higher limits give the agent more chances but take longer.

### Traditional LLM Benchmark

Run the traditional benchmark (without agent framework):

```bash
python benchmark_sonnet_only.py
```