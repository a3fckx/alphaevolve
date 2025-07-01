"""Utility functions for AlphaEvolve."""

import re
from typing import List, Tuple

EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"

def parse_evolve_blocks(code: str) -> Tuple[str, List[str]]:
    """
    Parses a string of code to find blocks marked for evolution.

    Args:
        code: The string containing the code.

    Returns:
        A tuple containing:
        - The code with evolve blocks replaced by placeholders (e.g., {{EVOLVE_BLOCK_0}}).
        - A list of the code blocks that can be evolved.
    """
    pattern = re.compile(f"{re.escape(EVOLVE_BLOCK_START)}(.*?){re.escape(EVOLVE_BLOCK_END)}", re.DOTALL)
    
    matches = list(re.finditer(pattern, code))
    blocks = [match.group(1).strip() for match in matches]
    
    # Replace blocks with placeholders in reverse order to not mess up indices
    for i, match in reversed(list(enumerate(matches))):
        start, end = match.span(0)
        placeholder = f"{{{{EVOLVE_BLOCK_{i}}}}}"
        code = code[:start] + placeholder + code[end:]
        
    return code, blocks

def reconstruct_code(template: str, evolved_blocks: List[str]) -> str:
    """
    Reconstructs the full code from a template and a list of evolved blocks.

    Args:
        template: The code template with placeholders.
        evolved_blocks: The list of evolved code blocks.

    Returns:
        The reconstructed full code.
    """
    reconstructed = template
    for i, block in enumerate(evolved_blocks):
        placeholder = f"{{{{EVOLVE_BLOCK_{i}}}}}"
        # Add the start/end markers back for clarity and consistency
        block_with_markers = f"{EVOLVE_BLOCK_START}\n{block}\n{EVOLVE_BLOCK_END}"
        reconstructed = reconstructed.replace(placeholder, block_with_markers)
    return reconstructed
