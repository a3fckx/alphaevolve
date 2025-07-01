"""Generates prompts for the LLM based on the problem and evolution state."""

from dataclasses import dataclass
from typing import List, Optional

from .database import Program
from .utils import reconstruct_code


@dataclass
class Prompt:
    """Represents a prompt to be sent to the LLM."""
    system_prompt: str
    user_prompt: str


class PromptSampler:
    """Generates prompts for the LLM based on the problem and evolution state."""
    
    def __init__(self, problem_description: str, evaluation_criteria: str):
        """Initialize the prompt sampler."""
        self.problem_description = problem_description
        self.evaluation_criteria = evaluation_criteria

    def generate_initial_prompt(self) -> Prompt:
        """Generate the initial prompt to create the first version of the code."""
        system_prompt = (
            "You are an expert programmer tasked with solving a challenging problem. "
            "Your goal is to write a complete Python program that solves the problem described below. "
            "The program should be self-contained and all necessary logic should be implemented."
        )
        
        user_prompt = (
            f"# Problem Description\n"
            f"{self.problem_description}\n\n"
            f"# Evaluation Criteria\n"
            f"{self.evaluation_criteria}\n\n"
            f"Please provide a complete Python program that solves this problem. "
            f"Enclose the code in a single markdown code block (```python...```)."
        )
        
        return Prompt(system_prompt, user_prompt)

    def generate_block_evolution_prompt(
        self, 
        code_template: str,
        evolve_block_index: int,
        evolve_block_code: str,
        previous_attempts: Optional[List[dict]] = None
    ) -> Prompt:
        """
        Generate a prompt to evolve a specific block of code.

        Args:
            code_template: The program code with placeholders for evolvable blocks.
            evolve_block_index: The index of the block to evolve.
            evolve_block_code: The current code of the block to be evolved.
            previous_attempts: A list of previous attempts for this block and their scores.
        """
        system_prompt = (
            "You are an expert programmer and a creative problem solver. "
            "Your task is to improve a specific block of code within a larger program. "
            "You will be given the context of the full program, the specific block to improve, and the evaluation criteria. "
            "Your goal is to rewrite the code block to better solve the problem. "
            "Do not just make trivial changes. Think of novel algorithms and creative solutions."
        )

        # Show the full program context with a marker for the block being evolved
        context_blocks = [f"{{{{EVOLVE_BLOCK_{i}}}}}" for i in range(code_template.count("EVOLVE_BLOCK"))]
        context_blocks[evolve_block_index] = f"\n# --- THIS IS THE BLOCK TO EVOLVE ---\n{evolve_block_code}\n# --- END OF BLOCK TO EVOLVE ---\n"
        full_context_code = reconstruct_code(code_template, context_blocks)

        history_section = ""
        if previous_attempts:
            history_section = "\n# Previous Attempts for this Block\n"
            for attempt in previous_attempts:
                history_section += f"- Score: {attempt['score']:.4f}\n"
                history_section += f"  ```python\n{attempt['code']}\n```\n"

        user_prompt = (
            f"# Problem Description\n"
            f"{self.problem_description}\n\n"
            f"# Evaluation Criteria\n"
            f"{self.evaluation_criteria}\n\n"
            f"# Full Program Context\n"
            f"Here is the full program. You must improve the block marked for evolution.\n"
            f"```python\n{full_context_code}\n```\n\n"
            f"# Current Code Block to Evolve\n"
            f"This is the specific code block you must rewrite and improve:\n"
            f"```python\n{evolve_block_code}\n```\n"
            f"{history_section}"
            f"Please provide only the new, improved code for this block. "
            f"Do not include the surrounding markers or the rest of the program. "
            f"Enclose your new code block in a single markdown code block (```python...```)."
        )

        return Prompt(system_prompt, user_prompt)