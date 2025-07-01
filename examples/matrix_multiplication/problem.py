"""Matrix multiplication optimization problem for AlphaEvolve."""

import asyncio
import math
from typing import List, Tuple, Dict, Any

from src.alphaevolve import AlphaEvolve, AlphaEvolveConfig
from src.evaluator import EvaluationResult
from src.config_loader import load_config


# Problem definition
PROBLEM_ID = "matrix_multiplication"
PROBLEM_DESCRIPTION = """
Optimize matrix multiplication for large matrices to achieve the fastest possible execution time. 
The goal is to implement an efficient algorithm that can handle matrices of size up to 1000x1000 with minimal computation time.
Your solution should focus on:
1. Reducing the number of arithmetic operations where possible (e.g., using Strassen's algorithm or other techniques)
2. Optimizing memory access patterns to improve cache efficiency
3. Leveraging parallelism if feasible within Python's standard library constraints

Your code should define a function called `matrix_multiply(A, B)` that takes two matrices as input and returns:
- The resulting matrix after multiplication

The function should minimize execution time while ensuring correctness of the result.

IMPORTANT: Use only Python standard library. Do NOT import numpy, scipy, or any external packages.
Only use: math, random, time, and other standard library modules.
"""

EVALUATION_CRITERIA = """
The score is based on a multi-objective evaluation:
1. Primary: Execution time for multiplying two 500x500 matrices (faster is better, accounts for 70% of the score)
2. Correctness: The result must be accurate compared to a baseline implementation (mandatory for a valid solution)
3. Secondary: Memory efficiency (measured by peak memory usage if possible, accounts for 20% of the score)
4. Tertiary: Code simplicity and readability (as a tiebreaker, accounts for 10% of the score)
"""


async def evaluate_matrix_multiplication(code: str, test_cases: Dict[str, Any] = None) -> EvaluationResult:
    """Custom evaluator for matrix multiplication optimization."""
    import subprocess
    import tempfile
    import json
    import os
    import time
    import random
    
    # Create evaluation script with embedded evaluator functions
    eval_script = f'''
import json
import time
import random
import sys

def generate_matrix(n):
    """Generate a random nxn matrix with float values."""
    return [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]

def baseline_matrix_multiply(A, B):
    """Baseline implementation for correctness checking."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    result = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    return result

def check_correctness(result, expected, tolerance=1e-6):
    """Check if the result matrix matches the expected matrix within tolerance."""
    if len(result) != len(expected) or len(result[0]) != len(expected[0]):
        return False, f"Dimension mismatch: got {len(result)}x{len(result[0])}, expected {len(expected)}x{len(expected[0])}"
    for i in range(len(result)):
        for j in range(len(result[0])):
            if abs(result[i][j] - expected[i][j]) > tolerance:
                return False, f"Value mismatch at ({i},{j}): got {result[i][j]}, expected {expected[i][j]}"
    return True, "Correct result"

# User code
{code}

try:
    # Generate test matrices
    n = 500
    A = generate_matrix(n)
    B = generate_matrix(n)
    
    # Compute expected result using baseline
    start_time_baseline = time.time()
    expected = baseline_matrix_multiply(A, B)
    baseline_time = time.time() - start_time_baseline
    
    # Run the user's implementation
    start_time_user = time.time()
    result = matrix_multiply(A, B)
    execution_time = time.time() - start_time_user
    
    # Check correctness
    correct, message = check_correctness(result, expected)
    
    if correct:
        # Score based on performance relative to baseline
        performance_ratio = baseline_time / execution_time if execution_time > 0 else 1.0
        score = min(1.0, 0.5 + 0.5 * performance_ratio)
        
        metrics = {{
            'execution_time': execution_time,
            'baseline_time': baseline_time,
            'performance_ratio': performance_ratio,
            'correct': True,
            'message': message
        }}
    else:
        score = 0.0
        metrics = {{
            'execution_time': execution_time,
            'baseline_time': baseline_time,
            'correct': False,
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
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return EvaluationResult(
                score=0.0,
                metrics={'error': 'Evaluation timed out after 120 seconds'},
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


async def run_matrix_multiplication(config_path: str = None):
    """Run the matrix multiplication optimization evolution."""
    # Load config from file
    config = load_config(config_path)
    
    # Override some settings specific to matrix multiplication if needed
    # config.population_size = 20  # Smaller population for faster iterations
    # config.generations = 50  # Fewer generations for testing
    # config.evaluation_timeout = 120  # More time for evaluation of large matrices
    # config.temperature = 0.6  # Moderate temperature for balanced creativity
    
    evolve = AlphaEvolve(
        problem_id=PROBLEM_ID,
        problem_description=PROBLEM_DESCRIPTION,
        evaluation_criteria=EVALUATION_CRITERIA,
        problem_type="optimization",
        custom_evaluator=evaluate_matrix_multiplication,
        config=config
    )
    
    best_program = await evolve.run()
    
    print(f"\nBest program achieved execution time: {best_program.metrics.get('execution_time', 0):.2f} seconds")
    print(f"Score: {best_program.score:.4f}")
    print(f"Performance ratio compared to baseline: {best_program.metrics.get('performance_ratio', 0):.2f}")
    print("\nCode:")
    print(best_program.code)
    
    return best_program


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_matrix_multiplication(config_path))
