# Understanding AlphaEvolve Evaluation

## How Evaluation Works

The evaluation system is the heart of AlphaEvolve. It determines which programs survive and evolve. Here's how it works:

### 1. Code Generation
Claude generates Python code based on the problem description and evolution prompts.

### 2. Sandboxed Execution
The generated code runs in a sandboxed subprocess with:
- Time limits (default: 30 seconds)
- Memory limits (default: 512MB)
- No network access
- Limited system calls

### 3. Problem-Specific Evaluation
Each problem defines its own evaluation logic:

## Circle Packing Example

```python
# Generated code must define:
def pack_circles(n):
    # Algorithm to position circles
    centers = [(x1, y1), (x2, y2), ...]  # n center positions
    radius = 0.095  # Radius achieved
    return centers, radius
```

### Evaluation Steps:

1. **Constraint Validation**
   - All circles inside unit square?
   - No overlaps between circles?
   - Correct number of circles?

2. **Performance Scoring**
   - Larger radius = better score
   - Score = radius / known_best_radius
   - Additional metrics: coverage, symmetry

3. **Results**
   ```json
   {
     "score": 0.95,
     "metrics": {
       "radius": 0.0975,
       "valid": true,
       "coverage": 0.75,
       "execution_time": 0.123
     }
   }
   ```

## Function Optimization Example

```python
# Generated code must define:
def optimize():
    # Algorithm to find minimum
    best_x = [0.001, -0.002, ...]  # 10D vector
    best_value = 0.05  # Function value
    return best_x, best_value
```

### Evaluation Steps:

1. **Solution Validation**
   - Correct dimensions (10)?
   - Within bounds [-5.12, 5.12]?

2. **Performance Scoring**
   - Closer to global minimum (0) = better
   - Logarithmic scale for discrimination
   - Execution time as tiebreaker

3. **Metrics Tracked**
   - Function value achieved
   - Distance from optimum
   - Convergence quality
   - Algorithm efficiency

## Custom Evaluators

You can create custom evaluators for any problem:

```python
async def evaluate_my_problem(code: str, test_cases: dict) -> EvaluationResult:
    # 1. Create evaluation script
    eval_script = f'''
{code}

# Your evaluation logic here
result = your_function()
score = calculate_score(result)
metrics = collect_metrics(result)
'''
    
    # 2. Execute in sandbox
    process = await asyncio.create_subprocess_exec(...)
    
    # 3. Return results
    return EvaluationResult(
        score=score,
        metrics=metrics,
        success=True
    )
```

## Scoring Philosophy

Good evaluation functions should:

1. **Be Deterministic**: Same code → same score
2. **Provide Gradients**: Small improvements → small score increases
3. **Avoid Plateaus**: Differentiate between similar solutions
4. **Balance Multiple Objectives**: Speed vs. quality trade-offs
5. **Handle Edge Cases**: Invalid solutions get score 0

## Evolution Strategies

Based on evaluation scores, AlphaEvolve decides:
- Which programs to keep (elite selection)
- Which to use as parents (tournament selection)
- How to generate new variants (mutation/crossover)
- When to explore vs. exploit

The evaluation system drives the entire evolutionary process!