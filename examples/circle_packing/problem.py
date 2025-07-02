"""Circle packing problem for AlphaEvolve."""

import asyncio
import logging
from typing import Dict, Any

from src.alphaevolve import AlphaEvolve
from examples.circle_packing.evaluator import evaluate_circle_packing


# Problem definition
PROBLEM_ID = "circle_packing_26"
PROBLEM_DESCRIPTION = """
This problem challenges the model to find the densest packing of N circles of varying radii inside a unit square.
This is a classic and difficult optimization problem, especially as N increases.
The goal is to maximize the sum of the radii of the circles.

The evolvable code block is the `pack_circles(n)` function, which should implement an algorithm to find the circle centers and radii.

Pack 26 circles of varying radii into a unit square (1x1) to solve an advanced optimization challenge such that:
1. No circles overlap (the distance between centers must be at least the sum of their radii)
2. All circles are completely inside the square (centers must be within their radius distance from all edges)
3. The sum of the radii of the circles is maximized to achieve the densest possible packing
4. The solution should aim for symmetry and uniform distribution as a secondary objective

Your code should define a function called `pack_circles(n)` that returns:
- A list of (x, y, radius) tuples for the center and radius of each circle
- The sum of the radii of the circles

The function should maximize the sum of radii while satisfying all constraints and consider symmetry for tie-breaking scenarios.

IMPORTANT: Use only Python standard library. Do NOT import numpy, scipy, or any external packages.
Only use: math, random, itertools, and other standard library modules.
"""

EVALUATION_CRITERIA = """
The score is based on a multi-objective evaluation:
1. Primary: The sum of radii achieved (larger is better, accounts for 80% of the score)
2. Constraint satisfaction (no overlaps, all circles inside square; mandatory for a valid solution)
3. Secondary: Symmetry and packing efficiency (measured by center of mass and coverage, accounts for 15% of the score)
4. Tertiary: Execution time (faster is better as a tiebreaker, accounts for 5% of the score)
"""


async def main():
    """Main function to run the evolution."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Read the initial code from the separate file, ensuring proper formatting
    initial_seed_code = ""
    with open("examples/circle_packing/initial_code.py", "r") as f:
        lines = f.readlines()
        # Remove any trailing whitespace or hidden characters and normalize line endings to \n
        initial_seed_code = "\n".join(line.rstrip() for line in lines)
    
    # Debug: Print the initial code to verify its content
    logger.info("Initial code being passed to AlphaEvolve:")
    logger.info(initial_seed_code)

    evolver = AlphaEvolve(
        problem_id="circle_packing",
        problem_description=PROBLEM_DESCRIPTION,
        evaluation_criteria=EVALUATION_CRITERIA,
        problem_type="optimization",
        custom_evaluator=evaluate_circle_packing,
        initial_code=initial_seed_code,
        test_cases=None
    )
    
    best_program = await evolver.run()
    
    if best_program:
        logger.info(f"Best program found with score: {best_program.score}")
        print(best_program.code)
    else:
        logger.info("Evolution finished without a valid program.")

if __name__ == "__main__":
    asyncio.run(main())
