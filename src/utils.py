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
    blocks = [match.group(1) for match in matches]  # Do not strip whitespace to preserve indentation
    
    # Replace blocks with placeholders in reverse order to not mess up indices
    for i, match in reversed(list(enumerate(matches))):
        start, end = match.span(0)
        placeholder = f"{{{{EVOLVE_BLOCK_{i}}}}}"
        code = code[:start] + placeholder + code[end:]
        
    return code, blocks

def reconstruct_code(template: str, evolved_blocks: List[str]) -> str:
    """
    Reconstructs the full code from a template and a list of evolved blocks, preserving relative indentation.

    Args:
        template: The code template with placeholders.
        evolved_blocks: The list of evolved code blocks.

    Returns:
        The reconstructed full code.
    """
    reconstructed = template
    for i, block in enumerate(evolved_blocks):
        placeholder = f"{{{{EVOLVE_BLOCK_{i}}}}}"
        # Use a base indentation level of 4 spaces for the block content to match typical Python function body
        base_indent = 4
        indented_block = []
        lines = block.split('\n')
        if lines:
            # Calculate the base indentation from the first non-empty line
            first_line_indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines:
                if line.strip():  # Only process non-empty lines
                    # Calculate the relative indentation for this line
                    line_indent = len(line) - len(line.lstrip())
                    # Apply base indent plus relative indent beyond the first line's indent
                    total_indent = base_indent + (line_indent - first_line_indent if line_indent >= first_line_indent else 0)
                    indented_block.append(' ' * total_indent + line.lstrip())
                else:
                    indented_block.append('')
        block_content = '\n'.join(indented_block)
        block_with_markers = f"{EVOLVE_BLOCK_START}\n{block_content}\n{' ' * base_indent}{EVOLVE_BLOCK_END}"
        reconstructed = reconstructed.replace(placeholder, block_with_markers)
    return reconstructed
