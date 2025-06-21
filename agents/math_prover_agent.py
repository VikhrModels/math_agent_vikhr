"""
Math Prover Agent - Main agent for mathematical theorem proving.

This module contains the main agent logic for automatically generating proofs
for mathematical theorems using Large Language Models.
"""

import os
import logging
from typing import List, Dict, Any, Optional

# Import smolagents
from smolagents import CodeAgent, ToolCallingAgent
from smolagents.models import OpenAIServerModel

# Import configuration
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    validate_config,
    get_model_for_environment,
    get_subset_size_for_environment,
    LOG_FORMAT
)

# Import tools
from .tools import logger

class MathProverAgent:
    """
    Main agent for mathematical theorem proving.
    
    This agent uses Large Language Models to automatically generate proofs
    for mathematical theorems written in Lean 3.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        agent_type: str = "code",  # "code" or "tool_calling"
        tools: Optional[List] = None,
        add_base_tools: bool = True,
        **kwargs
    ):
        """
        Initialize the Math Prover Agent.
        
        Args:
            model_name: Name of the model to use. If None, uses default from config.
            agent_type: Type of agent ("code" for CodeAgent, "tool_calling" for ToolCallingAgent)
            tools: List of tools to provide to the agent
            add_base_tools: Whether to add base tools from smolagents
            **kwargs: Additional arguments to pass to the agent
        """
        # Validate configuration
        validate_config()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT
        )
        
        # Initialize model
        self.model_name = model_name or get_model_for_environment()
        self.model = self._initialize_model()
        
        # Initialize agent
        self.agent_type = agent_type
        self.tools = tools or []
        self.add_base_tools = add_base_tools
        self.agent = self._initialize_agent(**kwargs)
        
        logger.info(f"MathProverAgent initialized with model: {self.model_name}")
        logger.info(f"Agent type: {self.agent_type}")
    
    def _initialize_model(self) -> OpenAIServerModel:
        """Initialize the OpenAIServerModel for OpenRouter."""
        return OpenAIServerModel(
            model_id=self.model_name,
            api_base=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY
        )
    
    def _initialize_agent(self, **kwargs) -> CodeAgent | ToolCallingAgent:
        """Initialize the appropriate agent type."""
        if self.agent_type == "code":
            return CodeAgent(
                tools=self.tools,
                model=self.model,
                add_base_tools=self.add_base_tools,
                **kwargs
            )
        elif self.agent_type == "tool_calling":
            return ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                add_base_tools=self.add_base_tools,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def run(self, task: str, reset: bool = True) -> str:
        """
        Run the agent on a given task.
        
        Args:
            task: The task to execute
            reset: Whether to reset the agent's memory before running
            
        Returns:
            The agent's response
        """
        logger.info(f"Running task: {task}")
        try:
            response = self.agent.run(task, reset=reset)
            logger.info("Task completed successfully")
            return response
        except Exception as e:
            logger.error(f"Error running task: {e}")
            raise
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get the agent's execution logs."""
        return self.agent.logs
    
    def write_memory_to_messages(self) -> List[Dict[str, Any]]:
        """Write the agent's memory as chat messages."""
        return self.agent.write_memory_to_messages()
    
    def interrupt(self) -> None:
        """Interrupt the agent's execution."""
        self.agent.interrupt()

# Convenience function for quick initialization
def create_math_prover_agent(
    model_name: Optional[str] = None,
    agent_type: str = "code",
    **kwargs
) -> MathProverAgent:
    """
    Create a MathProverAgent with default settings.
    
    Args:
        model_name: Name of the model to use
        agent_type: Type of agent ("code" or "tool_calling")
        **kwargs: Additional arguments for the agent
        
    Returns:
        Initialized MathProverAgent
    """
    return MathProverAgent(
        model_name=model_name,
        agent_type=agent_type,
        **kwargs
    )

# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_math_prover_agent()
    
    # Run a simple task
    print("Sending a task to the agent...")
    response = agent.run("What is the 10th prime number? Use code to find it and print the result.")
    
    print("\nAgent Response:")
    print(response)
