"""Custom evaluator for circle packing problem."""

import asyncio
import json
import os
import tempfile
from typing import Dict, Any

from src.evaluator import EvaluationResult


async def evaluate_circle_packing(code: str, test_cases: Dict[str, Any] = None) -> EvaluationResult:
    """Custom evaluator for circle packing with varying radii."""
    eval_script = f'''
import json
import time
import math
import traceback

# --- Helper functions for evaluation ---
def validate_circle_packing(circles, n):
    if not isinstance(circles, list) or not all(isinstance(c, tuple) and len(c) == 3 for c in circles):
        return False, "Circles must be a list of (x, y, radius) tuples."
    if len(circles) != n:
        return False, f"Expected {{n}} circles, got {{len(circles)}}."

    for i, (x, y, radius) in enumerate(circles):
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(radius, (int, float))):
            return False, f"Circle {{i}} coordinates and radius must be numbers."
        if not (radius > 0):
            return False, f"Circle {{i}} radius must be positive."
        if not (radius <= x <= 1 - radius and radius <= y <= 1 - radius):
            return False, f"Circle {{i}} is outside the unit square."

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2
            min_dist = circles[i][2] + circles[j][2]
            if dist_sq < min_dist**2 - 1e-9:
                return False, f"Circles {{i}} and {{j}} overlap."
    return True, "All constraints satisfied."

def score_circle_packing(sum_radii, n=26):
    # Scoring based on the sum of radii of circles.
    # The theoretical maximum sum of radii is difficult to compute exactly,
    # but we can use a reference value based on known packings or approximations.
    reference_sum_radii = 2.55  # Approximate reference for n=26 based on equal circles optimal packing
    score = sum_radii / reference_sum_radii if reference_sum_radii > 0 else 0
    return min(score, 1.0)  # Cap score at 1.0

# --- User code ---
{code}

# --- Debug the code being evaluated ---
with open("debug_code.py", "w") as f:
    f.write(code)

# --- Execution and Evaluation ---
try:
    start_time = time.time()
    # The user's code is expected to have a function `pack_circles(n)`
    circles, sum_radii = pack_circles(26)
    execution_time = time.time() - start_time

    valid, message = validate_circle_packing(circles, 26)

    if valid:
        score = score_circle_packing(sum_radii, 26)
        metrics = {{
            'sum_radii': sum_radii,
            'execution_time': execution_time,
            'valid': True,
            'message': message
        }}
    else:
        score = 0.0
        metrics = {{
            'sum_radii': sum_radii,
            'execution_time': execution_time,
            'valid': False,
            'message': message
        }}

    print(json.dumps({{'fitness': score, 'details': metrics}}))

except Exception as e:
    print(json.dumps({{
        'fitness': 0.0, 
        'details': {{
            'error': str(e),
            'traceback': traceback.format_exc()
        }}
    }}))
'''
    try:
        # Using a temporary file to run the script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(eval_script)
            script_path = f.name

        process = await asyncio.create_subprocess_exec(
            'python', script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if process.returncode == 0:
            result = json.loads(stdout.decode())
            eval_result = EvaluationResult(
                fitness=result.get('fitness'),
                details=result.get('details'),
                error=None
            )
            print(f"Evaluation Result: Fitness={eval_result.fitness}, Details={eval_result.details}")
            return eval_result
        else:
            return EvaluationResult(fitness=0.0, details={}, error=stderr.decode())

    except Exception as e:
        return EvaluationResult(fitness=0.0, details={}, error=str(e))
    finally:
        if 'script_path' in locals() and os.path.exists(script_path):
            os.unlink(script_path)
