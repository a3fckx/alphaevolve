"""Circle packing problem for AlphaEvolve."""

import asyncio
import math
from typing import List, Tuple, Dict, Any

from src.alphaevolve import AlphaEvolve, AlphaEvolveConfig
from src.evaluator import EvaluationResult


# Problem definition
PROBLEM_ID = "circle_packing_26"
PROBLEM_DESCRIPTION = """
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


async def run_circle_packing(config_path: str = None):
    """Run the circle packing evolution."""
    # Configuration is loaded directly in AlphaEvolve, pass the path if needed
    evolve = AlphaEvolve(
        problem_id=PROBLEM_ID,
        problem_description=PROBLEM_DESCRIPTION,
        evaluation_criteria=EVALUATION_CRITERIA,
        problem_type="optimization",
        custom_evaluator=evaluate_circle_packing,
        config_path=config_path if config_path else "config/config.yaml"
    )
    
    best_program = await evolve.run()
    
    print(f"\nBest program achieved radius: {best_program.metrics.get('radius', 0):.6f}")
    print(f"Score: {best_program.score:.4f}")
    print("\nCode:")
    print(best_program.code)
    
    return best_program


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_circle_packing(config_path))
