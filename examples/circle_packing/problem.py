"""Circle packing problem for AlphaEvolve."""

import asyncio
import logging
import math
from typing import List, Tuple, Dict, Any

from src.alphaevolve import AlphaEvolve, AlphaEvolveConfig
from src.evaluator import EvaluationResult


# Problem definition
PROBLEM_ID = "circle_packing_26"
PROBLEM_DESCRIPTION = """
This problem challenges the model to find the densest packing of N equal and non-overlapping circles inside a unit square.
This is a classic and difficult optimization problem, especially as N increases.
The goal is to maximize the radius of the circles.

The evolvable code block is the `pack_circles(n)` function, which should implement an algorithm to find the circle centers.

Note: For n=26, the SOTA was 2.634, and a previous version of AlphaEvolve improved it to 2.635.
For n=36, the SOTA was 2.936, and AlphaEvolve improved it to 2.937.

Pack 26 circles of equal radius into a unit square (1x1) to solve an advanced optimization challenge such that:
1. No circles overlap (the distance between centers must be at least twice the radius)
2. All circles are completely inside the square (centers must be within radius distance from all edges)
3. The radius of the circles is maximized to achieve the densest possible packing
4. The solution should aim for symmetry and uniform distribution as a secondary objective

Your code should define a function called `pack_circles(n)` that returns:
- A list of (x, y) coordinates for the center of each circle
- The radius of the circles

The function should maximize the radius while satisfying all constraints and consider symmetry for tie-breaking scenarios.

IMPORTANT: Use only Python standard library. Do NOT import numpy, scipy, or any external packages.
Only use: math, random, itertools, and other standard library modules.
"""

EVALUATION_CRITERIA = """
The score is based on a multi-objective evaluation:
1. Primary: The radius achieved (larger is better, accounts for 80% of the score)
2. Constraint satisfaction (no overlaps, all circles inside square; mandatory for a valid solution)
3. Secondary: Symmetry and packing efficiency (measured by center of mass and coverage, accounts for 15% of the score)
4. Tertiary: Execution time (faster is better as a tiebreaker, accounts for 5% of the score)
"""


async def evaluate_circle_packing(code: str, test_cases: Dict[str, Any] = None) -> EvaluationResult:
    """Custom evaluator for circle packing."""
    import subprocess
    import tempfile
    import json
    import os
    
    # Create evaluation script with embedded evaluator functions
    eval_script = f'''
import json
import time
import math

def validate_circle_packing(centers, radius, n):
    """Validate that a circle packing solution satisfies all constraints."""
    # Check correct number of circles
    if len(centers) != n:
        return False, f"Expected {{n}} circles, got {{len(centers)}}"
    
    # Check if all circles are inside the unit square
    for i, (x, y) in enumerate(centers):
        if x - radius < 0 or x + radius > 1:
            return False, f"Circle {{i}} extends outside square (x-axis): x={{x}}, r={{radius}}"
        if y - radius < 0 or y + radius > 1:
            return False, f"Circle {{i}} extends outside square (y-axis): y={{y}}, r={{radius}}"
    
    # Check for overlaps between circles
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = centers[i]
            x2, y2 = centers[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            min_distance = 2 * radius
            
            if distance < min_distance - 1e-9:  # Small tolerance for floating point
                return False, f"Circles {{i}} and {{j}} overlap: distance={{distance:.6f}}, required={{min_distance:.6f}}"
    
    return True, "All constraints satisfied"

def score_circle_packing(radius, n=26):
    """Score a circle packing solution based on achieved radius."""
    # Known best results for reference (approximate)
    known_best = {{
        25: 0.1,
        26: 0.0979,
        27: 0.0962,
        36: 0.0833,
        49: 0.0714
    }}
    
    reference = known_best.get(n, 0.1)
    
    # Score based on how close we are to known best
    ratio = radius / reference
    
    if ratio >= 1.0:
        return 1.0  # Perfect or better than known best
    elif ratio >= 0.99:
        return 0.9 + 0.1 * (ratio - 0.99) / 0.01
    elif ratio >= 0.95:
        return 0.7 + 0.2 * (ratio - 0.95) / 0.04
    elif ratio >= 0.90:
        return 0.5 + 0.2 * (ratio - 0.90) / 0.05
    else:
        return 0.5 * ratio / 0.90

def evaluate_packing_metrics(centers, radius):
    """Calculate additional metrics for a packing solution."""
    n = len(centers)
    
    # Calculate distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = centers[i]
            x2, y2 = centers[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            distances.append(dist)
    
    # Coverage calculation
    circle_area = math.pi * radius * radius
    total_circle_area = n * circle_area
    square_area = 1.0
    coverage = total_circle_area / square_area
    
    # Simple symmetry score based on center of mass
    cx = sum(x for x, y in centers) / n
    cy = sum(y for x, y in centers) / n
    symmetry_score = 1.0 - 2 * math.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    
    return {{
        'min_distance': min(distances) if distances else 0,
        'avg_distance': sum(distances) / len(distances) if distances else 0,
        'coverage': coverage,
        'symmetry_score': max(0, symmetry_score),
        'packing_efficiency': coverage / (math.pi / 4)
    }}

# User code
{code}

try:
    # Run the packing function
    start_time = time.time()
    centers, radius = pack_circles(26)
    execution_time = time.time() - start_time
    
    # Validate solution
    valid, message = validate_circle_packing(centers, radius, 26)
    
    if valid:
        # Score the solution
        score = score_circle_packing(radius, 26)
        
        # Get additional metrics
        metrics = evaluate_packing_metrics(centers, radius)
        metrics.update({{
            'radius': radius,
            'execution_time': execution_time,
            'valid': True,
            'message': message
        }})
    else:
        score = 0.0
        metrics = {{
            'radius': radius if 'radius' in locals() else 0,
            'execution_time': execution_time,
            'valid': False,
            'message': message
        }}
    
    print(json.dumps({{'score': score, 'metrics': metrics}}))
    
except Exception as e:
    import traceback
    print(json.dumps({{
        'score': 0.0, 
        'metrics': {{
            'error': str(e),
            'traceback': traceback.format_exc()
        }}
    }}))
'''
    
    # Write to temporary file and execute
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(eval_script)
        script_path = f.name
    
    try:
        process = await asyncio.create_subprocess_exec(
            'python', script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return EvaluationResult(
                score=0.0,
                metrics={'error': 'Evaluation timed out after 30 seconds'},
                success=False,
                error="Evaluation timed out - possible infinite loop or very slow algorithm"
            )
        
        if process.returncode == 0:
            try:
                result = json.loads(stdout.decode().strip().split('\n')[-1])
                return EvaluationResult(
                    score=result['score'],
                    metrics=result['metrics'],
                    success=True
                )
            except Exception as e:
                return EvaluationResult(
                    score=0.0,
                    metrics={'error': f'Failed to parse results: {str(e)}', 'stdout': stdout.decode(), 'stderr': stderr.decode()},
                    success=False,
                    error=f"Parse error: {e}\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
                )
        else:
            return EvaluationResult(
                score=0.0,
                metrics={'error': 'Execution failed'},
                success=False,
                error=stderr.decode()
            )
    finally:
        os.unlink(script_path)


async def main():
    """Main function to run the evolution."""
    logging.basicConfig(level=logging.INFO)
    
    # The initial code is now passed directly to the constructor
    evolver = AlphaEvolve(
        problem_id="circle_packing",
        problem_description=PROBLEM_DESCRIPTION,
        evaluation_criteria=EVALUATION_CRITERIA,
        initial_code="""
def pack_circles(n):
    # Your implementation here
    return [], 0.0
""",
        test_cases=None
    )
    
    best_program = await evolver.run()
    
    if best_program:
        logger.info(f"Best program found with score: {best_program.score}")
        # You can save or inspect the best program's code
        # print(best_program.code)
    else:
        logger.info("Evolution finished without a valid program.")

if __name__ == "__main__":
    import sys
    asyncio.run(main())


def pack_circles(n):
    """
    Packs n circles of equal radius into a unit square, maximizing the radius.
    """
    if n <= 0:
        return [], 0.0

    # EVOLVE-BLOCK-START
    # Initial guess for radius - a very small value to start
    # The maximum possible radius for n=1 is 0.5
    # For n=26, it will be significantly smaller.
    # A simple grid packing of 5x5 would give a radius of 1/(2*5) = 0.1
    # So, let's start with a radius slightly larger than that for initial attempts.
    # We will use a binary search approach to find the maximum radius.

    low = 0.0
    # A safe upper bound for the radius is 0.5 (for n=1).
    # For n=26, it will be much smaller, but 0.5 is a valid upper bound.
    high = 0.5
    best_radius = 0.0
    best_centers = []

    # Number of iterations for the binary search. More iterations lead to higher precision.
    # 100 iterations are usually sufficient for good precision.
    num_binary_search_iterations = 100

    for _ in range(num_binary_search_iterations):
        mid_radius = (low + high) / 2.0
        if mid_radius == 0: # Avoid division by zero if high becomes very small
            break

        # Try to place n circles with the current mid_radius
        # We'll use a simple random placement strategy with some local optimization
        # to try and find a valid packing for the given radius.
        # If we can successfully place n circles, we try a larger radius.
        # If not, we try a smaller radius.

        # --- Attempt to place n circles with mid_radius ---
        # This is the core of the problem and the most challenging part.
        # For a general n, there's no simple analytical solution.
        # We'll use a heuristic approach: random placement with some checks.
        # A more robust solution would involve complex optimization algorithms (e.g., simulated annealing, genetic algorithms).
        # Given the constraints (no external libraries), we'll stick to a simpler heuristic.

        current_centers = []
        max_attempts_per_circle = 5000  # Limit attempts to place a single circle
        placement_successful = False

        # Try to place n circles
        for _ in range(n):
            placed = False
            for attempt in range(max_attempts_per_circle):
                # Generate a random center within the valid region for a circle of radius mid_radius
                x = random.uniform(mid_radius, 1.0 - mid_radius)
                y = random.uniform(mid_radius, 1.0 - mid_radius)
                new_center = (x, y)

                # Check if this new circle overlaps with existing ones or goes out of bounds
                if is_within_bounds(new_center, mid_radius) and is_valid_placement(current_centers + [new_center], mid_radius):
                    current_centers.append(new_center)
                    placed = True
                    break # Successfully placed this circle, move to the next

            if not placed:
                # Failed to place the current circle, so this radius is too large
                break

        if len(current_centers) == n:
            # Successfully placed all n circles with mid_radius
            # This radius is achievable, so we try for a larger one
            best_radius = mid_radius
            best_centers = current_centers
            low = mid_radius
        else:
            # Could not place all n circles with mid_radius
            # This radius is too large, so we try a smaller one
            high = mid_radius

    # After binary search, best_radius and best_centers hold the best found packing.
    # For n=26, the random placement might not be the most optimal or symmetrical.
    # We can try to improve the symmetry and distribution of the best found centers.

    # --- Post-processing for Symmetry and Distribution ---
    # This is a simple local search/adjustment.
    # We can try to move each center slightly to improve its position relative to others
    # and the boundaries, aiming for better symmetry.

    # A very basic approach: try to center the packing.
    if best_centers:
        avg_x = sum(c[0] for c in best_centers) / n
        avg_y = sum(c[1] for c in best_centers) / n
        offset_x = 0.5 - avg_x
        offset_y = 0.5 - avg_y

        adjusted_centers = []
        for cx, cy in best_centers:
            new_cx = cx + offset_x
            new_cy = cy + offset_y
            # Ensure adjusted centers are still within bounds for the radius
            # This might slightly reduce the radius if adjustment pushes them out
            # but it's a trade-off for symmetry.
            # For simplicity here, we'll assume the offset is small enough not to violate it severely
            # or that the binary search already found a radius that allows some flexibility.
            # A more robust solution would re-check validity after adjustment.
            adjusted_centers.append((new_cx, new_cy))
        
        # Re-check validity after adjustment, though it might be strict.
        # For this problem, we prioritize the radius found.
        # If adjustments break constraints, we might stick to the original best_centers.
        # For now, we'll return the adjusted centers if they are valid.

        is_adjusted_valid = True
        for i, center in enumerate(adjusted_centers):
            if not is_within_bounds(center, best_radius):
                is_adjusted_valid = False
                break
            for j in range(i):
                if distance(center, adjusted_centers[j]) < 2 * best_radius:
                    is_adjusted_valid = False
                    break
            if not is_adjusted_valid:
                break
        
        if is_adjusted_valid:
            best_centers = adjusted_centers
        # If adjustment made it invalid, we keep the original best_centers.
    # EVOLVE-BLOCK-END

    return best_centers, best_radius

# Example usage:
