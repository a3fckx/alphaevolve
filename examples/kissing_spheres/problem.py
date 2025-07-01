"""Kissing spheres problem for AlphaEvolve - 3D sphere packing."""

import asyncio
import math
from typing import List, Tuple, Dict, Any

from src.alphaevolve import AlphaEvolve, AlphaEvolveConfig
from src.evaluator import EvaluationResult


# Problem definition
PROBLEM_ID = "kissing_spheres_3d"
PROBLEM_DESCRIPTION = """
The Kissing Number Problem: Find the maximum number of non-overlapping unit spheres 
that can simultaneously touch a central unit sphere in 3D space, with potential exploration into higher dimensions.

Your code should define a function called `find_kissing_spheres()` that returns:
- A list of (x, y, z) coordinates for the center of each touching sphere (for 3D case)
- The number of spheres found

Requirements:
1. All spheres have radius 1.0
2. Each sphere must touch the central sphere at (0, 0, 0)
3. No two spheres can overlap (minimum distance between centers = 2.0)
4. All sphere centers must be exactly distance 2.0 from origin

The theoretical maximum for 3D is 12 spheres (proven by Newton/Gregory). AlphaEvolve aims to match or approach this maximum in 3D and explore novel configurations, potentially extending insights to higher dimensions as demonstrated in the technical report.

IMPORTANT: Use only Python standard library. Focus on finding valid configurations
that maximize the number of touching spheres.

Example structure:
```python
def find_kissing_spheres():
    # Your algorithm here
    centers = []  # List of (x, y, z) tuples
    
    # Generate sphere positions that satisfy constraints
    # ... your code ...
    
    return centers, len(centers)
```
"""

EVALUATION_CRITERIA = """
The score is based on:
1. Number of valid touching spheres (more is better, max theoretical = 12)
2. Constraint satisfaction (all spheres touch center, no overlaps)
3. Numerical accuracy of the configuration
"""


async def evaluate_kissing_spheres(code: str, test_cases: Dict[str, Any] = None) -> EvaluationResult:
    """Custom evaluator for kissing spheres problem."""
    import subprocess
    import tempfile
    import json
    import os
    
    # Create evaluation script
    eval_script = f'''
import json
import math

def validate_kissing_spheres(centers):
    """Validate that sphere configuration satisfies all constraints."""
    n = len(centers)
    
    if n == 0:
        return False, "No spheres found", 0
    
    # Check that all spheres touch the central sphere
    for i, (x, y, z) in enumerate(centers):
        distance = math.sqrt(x*x + y*y + z*z)
        if abs(distance - 2.0) > 1e-6:  # Should be exactly 2.0 (two radii)
            return False, f"Sphere {{i}} not touching center: distance={{distance:.6f}}", 0
    
    # Check for overlaps between spheres
    overlaps = 0
    min_allowed_distance = 2.0 - 1e-6  # Two radii minus small tolerance
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, z1 = centers[i]
            x2, y2, z2 = centers[j]
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            if distance < min_allowed_distance:
                overlaps += 1
    
    if overlaps > 0:
        return False, f"Found {{overlaps}} overlapping sphere pairs", n
    
    return True, "All constraints satisfied", n

def score_kissing_spheres(centers):
    """Score based on number of valid spheres."""
    valid, message, n = validate_kissing_spheres(centers)
    
    if not valid:
        # Partial credit for configurations with minor issues
        if "not touching" in message:
            return 0.1  # Very low score for incorrect distances
        elif "overlapping" in message:
            # Some credit based on number of spheres minus penalty for overlaps
            return min(0.5, n / 12.0 * 0.5)
        else:
            return 0.0
    
    # Score based on number of spheres (12 is theoretical maximum)
    if n >= 12:
        return 1.0  # Perfect score for achieving theoretical maximum
    elif n >= 10:
        return 0.8 + (n - 10) * 0.1  # Good score for 10-11 spheres
    elif n >= 8:
        return 0.6 + (n - 8) * 0.1   # Decent score for 8-9 spheres
    elif n >= 6:
        return 0.4 + (n - 6) * 0.1   # Moderate score for 6-7 spheres
    else:
        return n / 12.0 * 0.4         # Linear scaling for fewer spheres

def calculate_metrics(centers):
    """Calculate additional metrics for the configuration."""
    n = len(centers)
    if n == 0:
        return {{'num_spheres': 0}}
    
    # Check distances
    distances_from_center = []
    for x, y, z in centers:
        dist = math.sqrt(x*x + y*y + z*z)
        distances_from_center.append(dist)
    
    # Check pairwise distances
    pairwise_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, z1 = centers[i]
            x2, y2, z2 = centers[j]
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            pairwise_distances.append(dist)
    
    # Calculate symmetry metrics
    center_of_mass_x = sum(x for x, y, z in centers) / n
    center_of_mass_y = sum(y for x, y, z in centers) / n
    center_of_mass_z = sum(z for x, y, z in centers) / n
    com_distance = math.sqrt(center_of_mass_x**2 + center_of_mass_y**2 + center_of_mass_z**2)
    
    return {{
        'num_spheres': n,
        'avg_distance_from_center': sum(distances_from_center) / n if n > 0 else 0,
        'min_distance_from_center': min(distances_from_center) if distances_from_center else 0,
        'max_distance_from_center': max(distances_from_center) if distances_from_center else 0,
        'min_pairwise_distance': min(pairwise_distances) if pairwise_distances else 0,
        'avg_pairwise_distance': sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0,
        'center_of_mass_distance': com_distance,
        'symmetry_score': 1.0 - min(1.0, com_distance)  # Perfect symmetry = 1.0
    }}

# User code
{code}

try:
    # Run the function
    import time
    start_time = time.time()
    centers, num_spheres = find_kissing_spheres()
    execution_time = time.time() - start_time
    
    # Validate and score
    score = score_kissing_spheres(centers)
    metrics = calculate_metrics(centers)
    metrics['execution_time'] = execution_time
    
    # Check validity
    valid, message, _ = validate_kissing_spheres(centers)
    metrics['valid'] = valid
    metrics['message'] = message
    
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
                error="Evaluation timed out"
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
                    metrics={'error': f'Parse error: {str(e)}'},
                    success=False,
                    error=str(e)
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


async def run_kissing_spheres(config_path: str = None):
    """Run the kissing spheres evolution."""
    config = load_config(config_path)
    
    evolve = AlphaEvolve(
        problem_id=PROBLEM_ID,
        problem_description=PROBLEM_DESCRIPTION,
        evaluation_criteria=EVALUATION_CRITERIA,
        problem_type="optimization",
        custom_evaluator=evaluate_kissing_spheres,
        config=config
    )
    
    best_program = await evolve.run()
    
    if best_program:
        print(f"\nBest program found {best_program.metrics.get('num_spheres', 0)} kissing spheres")
        print(f"Score: {best_program.score:.4f}")
        print("\nCode:")
        print(best_program.code)
    
    return best_program


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_kissing_spheres(config_path))
