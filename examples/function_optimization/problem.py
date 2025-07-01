"""Function optimization problem for AlphaEvolve."""

import asyncio
import math
from typing import Dict, Any

from src.alphaevolve import AlphaEvolve, AlphaEvolveConfig
from src.evaluator import EvaluationResult


# Problem definition
PROBLEM_ID = "rastrigin_optimization"
PROBLEM_DESCRIPTION = """
Minimize the Rastrigin function in 10 dimensions, a benchmark for optimization algorithms.

The Rastrigin function is defined as:
f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i)) for i=1 to n

where n=10 and each x_i is in the range [-5.12, 5.12].

Your code should define a function called `optimize()` that returns:
- The best x vector found (list of 10 values)
- The function value at that point

The global minimum is at x = [0, 0, ..., 0] with f(x) = 0. AlphaEvolve aims to approach this minimum through innovative optimization strategies, potentially discovering novel algorithms that compete with traditional methods as demonstrated in the technical report.

IMPORTANT: Use only Python standard library. Focus on developing efficient optimization techniques to navigate high-dimensional search spaces with many local minima.
"""

EVALUATION_CRITERIA = """
The score is based on:
1. How close the function value is to the global minimum (0)
2. The quality of the optimization algorithm
3. Convergence speed
"""


async def evaluate_optimization(code: str, test_cases: Dict[str, Any] = None) -> EvaluationResult:
    """Custom evaluator for function optimization."""
    import subprocess
    import tempfile
    import json
    import os
    
    # Create evaluation script
    eval_script = f'''
import json
import math
import time
import random

def rastrigin(x):
    """The Rastrigin function to minimize."""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

{code}

try:
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run the optimization
    start_time = time.time()
    best_x, best_value = optimize()
    execution_time = time.time() - start_time
    
    # Verify the result
    actual_value = rastrigin(best_x)
    
    # Check if solution is valid
    if len(best_x) != 10:
        raise ValueError(f"Solution must have 10 dimensions, got {{len(best_x)}}")
    
    if any(abs(xi) > 5.12 for xi in best_x):
        raise ValueError("Solution components must be in [-5.12, 5.12]")
    
    # Score based on how close we are to global minimum
    # Use logarithmic scale for better discrimination
    if actual_value <= 0.01:
        score = 1.0
    elif actual_value <= 0.1:
        score = 0.9
    elif actual_value <= 1.0:
        score = 0.8 - 0.1 * math.log10(actual_value)
    elif actual_value <= 10.0:
        score = 0.6 - 0.1 * math.log10(actual_value)
    else:
        score = max(0.0, 0.4 - 0.05 * math.log10(actual_value))
    
    metrics = {{
        'function_value': actual_value,
        'reported_value': best_value,
        'solution': best_x,
        'execution_time': execution_time,
        'distance_from_optimum': math.sqrt(sum(xi**2 for xi in best_x))
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
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        
        if process.returncode == 0:
            try:
                result = json.loads(stdout.decode().strip().split('\n')[-1])
                return EvaluationResult(
                    score=result['score'],
                    metrics=result['metrics'],
                    success=True
                )
            except:
                return EvaluationResult(
                    score=0.0,
                    metrics={'error': 'Failed to parse results'},
                    success=False,
                    error=stdout.decode() + stderr.decode()
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


async def run_function_optimization(config_path: str = None):
    """Run the function optimization evolution."""
    # Load config from file
    config = load_config(config_path)
    
 
    evolve = AlphaEvolve(
        problem_id=PROBLEM_ID,
        problem_description=PROBLEM_DESCRIPTION,
        evaluation_criteria=EVALUATION_CRITERIA,
        problem_type="optimization",
        custom_evaluator=evaluate_optimization,
        config=config
    )
    
    best_program = await evolve.run()
    
    print(f"\nBest program achieved function value: {best_program.metrics.get('function_value', float('inf')):.6f}")
    print(f"Score: {best_program.score:.4f}")
    print(f"Distance from optimum: {best_program.metrics.get('distance_from_optimum', float('inf')):.6f}")
    print("\nCode:")
    print(best_program.code)
    
    return best_program


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_function_optimization(config_path))
