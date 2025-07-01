"""Code evaluation system with sandboxed execution."""

import ast
import asyncio
import subprocess
import tempfile
import os
import json
import time
import traceback
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import resource
import signal

import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of code evaluation."""
    score: float
    metrics: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time: Optional[float] = None


class CodeEvaluator:
    """Evaluates code in a sandboxed environment."""
    
    def __init__(self, 
                 timeout: int = 30,
                 memory_limit_mb: int = 512,
                 custom_evaluator: Optional[Callable] = None):
        """Initialize evaluator."""
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.custom_evaluator = custom_evaluator
    
    async def evaluate(self, 
                      code: str, 
                      problem_type: str,
                      test_cases: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate code and return results."""
        # First, check if code is syntactically valid
        try:
            ast.parse(code)
        except SyntaxError as e:
            return EvaluationResult(
                score=0.0,
                metrics={"syntax_error": str(e)},
                success=False,
                error=f"Syntax error: {e}"
            )
        
        # Use custom evaluator if provided
        if self.custom_evaluator:
            try:
                return await self._run_custom_evaluator(code, test_cases)
            except Exception as e:
                logger.error(f"Custom evaluator failed: {e}", exc_info=True)
                return EvaluationResult(
                    score=0.0,
                    metrics={"evaluator_error": str(e), "traceback": traceback.format_exc()},
                    success=False,
                    error=f"Evaluator error: {e}\n{traceback.format_exc()}"
                )
        
        # Default sandbox evaluation
        return await self._sandbox_evaluate(code, problem_type, test_cases)
    
    async def _sandbox_evaluate(self, 
                               code: str, 
                               problem_type: str,
                               test_cases: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate code in a sandboxed subprocess."""
        # Create evaluation script
        eval_script = self._create_evaluation_script(code, problem_type, test_cases)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(eval_script)
            script_path = f.name
        
        try:
            # Run in subprocess with resource limits
            start_time = time.time()
            process = await asyncio.create_subprocess_exec(
                'python', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=self._set_resource_limits if os.name != 'nt' else None
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout
                )
                execution_time = time.time() - start_time
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return EvaluationResult(
                    score=0.0,
                    metrics={"timeout": True, "execution_time": self.timeout},
                    success=False,
                    error=f"Execution timed out after {self.timeout} seconds"
                )
            
            # Parse results
            if process.returncode == 0:
                try:
                    # Results are output as JSON on the last line
                    output_lines = stdout.decode().strip().split('\n')
                    result_json = output_lines[-1]
                    result = json.loads(result_json)
                    
                    return EvaluationResult(
                        score=result['score'],
                        metrics=result['metrics'],
                        success=True,
                        stdout='\n'.join(output_lines[:-1]) if len(output_lines) > 1 else None,
                        execution_time=execution_time
                    )
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    return EvaluationResult(
                        score=0.0,
                        metrics={"parse_error": str(e)},
                        success=False,
                        error=f"Failed to parse evaluation results: {e}",
                        stdout=stdout.decode() if stdout else None,
                        stderr=stderr.decode() if stderr else None
                    )
            else:
                return EvaluationResult(
                    score=0.0,
                    metrics={"return_code": process.returncode},
                    success=False,
                    error=f"Execution failed with return code {process.returncode}",
                    stdout=stdout.decode() if stdout else None,
                    stderr=stderr.decode() if stderr else None
                )
                
        finally:
            # Clean up
            try:
                os.unlink(script_path)
            except:
                pass
    
    def _set_resource_limits(self):
        """Set resource limits for subprocess (Unix only)."""
        # Memory limit
        resource.setrlimit(resource.RLIMIT_AS, 
                         (self.memory_limit_mb * 1024 * 1024, 
                          self.memory_limit_mb * 1024 * 1024))
        
        # CPU time limit (backup for timeout)
        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
        
        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        
        # Limit number of processes
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    
    def _create_evaluation_script(self, 
                                 code: str, 
                                 problem_type: str,
                                 test_cases: Optional[Dict[str, Any]] = None) -> str:
        """Create evaluation script based on problem type."""
        if problem_type == "optimization":
            return self._create_optimization_eval_script(code, test_cases)
        elif problem_type == "algorithm":
            return self._create_algorithm_eval_script(code, test_cases)
        else:
            return self._create_generic_eval_script(code, test_cases)
    
    def _create_optimization_eval_script(self, code: str, test_cases: Dict[str, Any]) -> str:
        """Create evaluation script for optimization problems."""
        return f'''
import json
import time
import sys
import traceback

# User code
{code}

# Evaluation
try:
    metrics = {{}}
    
    # Run optimization function
    start_time = time.time()
    result = optimize()  # Assumes user code defines optimize()
    execution_time = time.time() - start_time
    
    # Calculate score (problem-specific)
    score = -result if isinstance(result, (int, float)) else 0.0
    
    metrics['result'] = result
    metrics['execution_time'] = execution_time
    
    # Output results as JSON
    print(json.dumps({{'score': score, 'metrics': metrics}}))
    
except Exception as e:
    print(json.dumps({{'score': 0.0, 'metrics': {{'error': str(e), 'traceback': traceback.format_exc()}}}}))
'''
    
    def _create_algorithm_eval_script(self, code: str, test_cases: Dict[str, Any]) -> str:
        """Create evaluation script for algorithm problems."""
        return f'''
import json
import time
import sys
import traceback

# User code
{code}

# Test cases
test_cases = {json.dumps(test_cases) if test_cases else '{}'}

# Evaluation
try:
    metrics = {{}}
    total_score = 0
    passed_tests = 0
    
    for test_name, test_data in test_cases.items():
        try:
            start_time = time.time()
            result = solution(test_data['input'])  # Assumes user code defines solution()
            execution_time = time.time() - start_time
            
            if result == test_data['expected']:
                passed_tests += 1
                test_score = 1.0 / len(test_cases)
                total_score += test_score
                metrics[test_name] = {{'passed': True, 'time': execution_time}}
            else:
                metrics[test_name] = {{'passed': False, 'expected': test_data['expected'], 'got': result}}
                
        except Exception as e:
            metrics[test_name] = {{'passed': False, 'error': str(e)}}
    
    metrics['passed_tests'] = passed_tests
    metrics['total_tests'] = len(test_cases)
    
    # Output results as JSON
    print(json.dumps({{'score': total_score, 'metrics': metrics}}))
    
except Exception as e:
    print(json.dumps({{'score': 0.0, 'metrics': {{'error': str(e), 'traceback': traceback.format_exc()}}}}))
'''
    
    def _create_generic_eval_script(self, code: str, test_cases: Dict[str, Any]) -> str:
        """Create generic evaluation script."""
        return f'''
import json
import time
import sys
import traceback

# User code
{code}

# Evaluation
try:
    metrics = {{}}
    
    # Run main function if it exists
    if 'main' in globals():
        start_time = time.time()
        result = main()
        execution_time = time.time() - start_time
        metrics['execution_time'] = execution_time
        metrics['result'] = str(result) if result is not None else None
        score = 1.0  # Generic success score
    else:
        score = 0.5  # Code runs but no main function
        metrics['warning'] = 'No main function found'
    
    # Output results as JSON
    print(json.dumps({{'score': score, 'metrics': metrics}}))
    
except Exception as e:
    print(json.dumps({{'score': 0.0, 'metrics': {{'error': str(e), 'traceback': traceback.format_exc()}}}}))
'''
    
    async def _run_custom_evaluator(self, 
                                   code: str, 
                                   test_cases: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Run custom evaluator function."""
        # Custom evaluator should be an async function that returns EvaluationResult
        return await self.custom_evaluator(code, test_cases)