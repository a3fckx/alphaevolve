# AlphaEvolve

An implementation of Google DeepMind's AlphaEvolve - an evolutionary coding agent that uses Large Language Models (via Gemini API) to iteratively generate, evaluate, and optimize code for algorithmic problems.

## Overview

AlphaEvolve uses evolutionary algorithms combined with LLMs to discover and optimize algorithms. The system maintains a population of candidate programs, evaluates their performance, and uses the Gemini API to generate improved versions through various evolution strategies.

### Key Features

- **Evolutionary Optimization**: Uses mutation, crossover, and exploration strategies
- **LLM-Powered Code Generation**: Leverages Gemini API for intelligent code modifications
- **Sandboxed Evaluation**: Safe execution environment with resource limits
- **Checkpoint/Resume**: Save and restore evolution progress
- **Island Model**: Parallel evolution with migration for diversity
- **Flexible Architecture**: Easy to add new problems and evaluation metrics

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Prompt Sampler  │────►│ Gemini API       │────►│ Code Generator  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲                                                 │
         │                                                 ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Evolution       │◄────│    Database      │◄────│   Evaluator     │
│ Strategy        │     │  (Programs)      │     │   (Sandbox)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Installation

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alphaevolve
```

2. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your-actual-api-key-here
```

3. Create required directories:
```bash
mkdir -p data logs checkpoints
```

### Local Installation

1. Install Python 3.11+

2. Create virtual environment (using uv):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

4. Set up environment:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

## Data Management

Data is stored locally in the following directories:
- `./data`: SQLite database with evolution history
- `./logs`: Application logs
- `./checkpoints`: Evolution checkpoints for resume capability

## Usage

### Command Line Interface

First, activate your virtual environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Run an example problem:
```bash
python -m src.main run --problem circle_packing
```

Resume from checkpoint:
```bash
python -m src.main run --problem circle_packing --resume
```

Use custom configuration:
```bash
python -m src.main run --problem function_optimization --config config/custom.yaml
```

List available problems:
```bash
python -m src.main list-problems
```

Generate example configuration:
```bash
python -m src.main generate-config --output config/example.yaml
```

**Note**: By default, AlphaEvolve uses `config/config.yaml` for configuration settings.

### Python API

```python
import asyncio
from src.alphaevolve import AlphaEvolve, AlphaEvolveConfig

async def run_evolution():
    config = AlphaEvolveConfig(
        population_size=50,
        generations=100,
        temperature=0.7
    )
    
    evolve = AlphaEvolve(
        problem_id="my_problem",
        problem_description="Solve X efficiently",
        evaluation_criteria="Minimize time complexity",
        config=config
    )
    
    best_program = await evolve.run()
    print(f"Best score: {best_program.score}")
    print(f"Code:\n{best_program.code}")

asyncio.run(run_evolution())
```

## Example Problems

### Circle Packing
Pack 26 equal circles in a unit square to maximize radius.

```bash
# Run from main module
python -m src.main run --problem circle_packing

# Or run directly
python examples/circle_packing/problem.py
```

### Function Optimization
Minimize the 10-dimensional Rastrigin function.

```bash
# Run from main module
python -m src.main run --problem function_optimization

# Or run directly
python examples/function_optimization/problem.py
```

### Kissing Spheres
Optimize the arrangement of spheres to maximize the number of kissing points.

```bash
# Run from main module
python -m src.main run --problem kissing_spheres

# Or run directly
python examples/kissing_spheres/problem.py
```

### Matrix Multiplication
Optimize matrix multiplication for large matrices to achieve the fastest execution time.

```bash
# Run from main module
python -m src.main run --problem matrix_multiplication

# Or run directly
python examples/matrix_multiplication/problem.py
```

## Creating Custom Problems

1. Define your problem:
```python
PROBLEM_DESCRIPTION = """
Your problem description here...
"""

EVALUATION_CRITERIA = """
How solutions will be scored...
"""
```

2. Implement custom evaluator:
```python
async def evaluate_my_problem(code: str, test_cases: dict) -> EvaluationResult:
    # Create evaluation script with the user's code
    eval_script = f'''
import json
import time

# Your evaluation functions here
def validate_solution(...):
    # Validation logic
    pass

def score_solution(...):
    # Scoring logic
    pass

# User's generated code
{code}

try:
    # Run the user's solution
    result = user_function()
    
    # Validate and score
    valid = validate_solution(result)
    score = score_solution(result) if valid else 0.0
    
    print(json.dumps({{'score': score, 'metrics': {...}}}))
except Exception as e:
    print(json.dumps({{'score': 0.0, 'metrics': {{'error': str(e)}}}}))
'''
    
    # Execute in subprocess and return results
    # See examples/circle_packing/problem.py for full implementation
```

3. Run evolution:
```python
evolve = AlphaEvolve(
    problem_id="my_problem",
    problem_description=PROBLEM_DESCRIPTION,
    evaluation_criteria=EVALUATION_CRITERIA,
    custom_evaluator=evaluate_my_problem
)
```

## Configuration

Default configuration file (`config/config.yaml`):
```yaml
evolution:
  population_size: 50
  generations: 100
  elite_size: 5
  mutation_rate: 0.8
  crossover_rate: 0.2
  exploration_rate: 0.1
  checkpoint_interval: 10
  
openai:
  model: "google/gemini-2.0-flash-exp:free"
  temperature: 0.7
  max_tokens: 8192
  
evaluation:
  timeout: 30
  memory_limit_mb: 512

database:
  url: "sqlite:///data/alphaevolve.db"

logging:
  level: "INFO"
  file: "logs/alphaevolve.log"
```

## Database Schema

The system uses SQLite to track:
- **Programs**: Generated code with scores and metrics
- **Evolution Runs**: Configuration and progress tracking
- **Checkpoints**: Population snapshots for resume capability

## Safety and Security

- Code evaluation runs in sandboxed subprocess
- Resource limits (CPU, memory, time)
- No network access during evaluation
- Syntax validation before execution

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/
```

Type checking:
```bash
mypy src/
```

## Performance Tips

1. **Population Size**: Larger populations explore more but cost more API calls
2. **Temperature**: Higher values increase creativity but may reduce consistency
3. **Elite Size**: Preserve best solutions while maintaining diversity
4. **Evaluation Timeout**: Balance between allowing complex solutions and efficiency

## Limitations

- Requires Gemini API access (costs may apply)
- Evaluation limited to Python code
- Sandbox restrictions may affect some algorithms
- Best for well-defined optimization problems

## Future Enhancements

- [ ] Multi-language support
- [ ] Distributed evaluation
- [ ] Web UI for monitoring
- [ ] More example problems
- [ ] Advanced visualization
- [ ] Multi-objective optimization

## References

Based on the paper: "AlphaEvolve: A coding agent for scientific and algorithmic discovery" by Google DeepMind (2025)

## License

[Your license here]

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
