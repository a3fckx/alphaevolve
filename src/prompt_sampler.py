"""Prompt generation for evolutionary code optimization."""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .database import Program


@dataclass
class EvolutionPrompt:
    """Represents a prompt for code evolution."""
    system_prompt: str
    user_prompt: str
    parent_program: Optional[Program] = None
    context_programs: List[Program] = None


class PromptSampler:
    """Generates prompts for code evolution using various strategies."""
    
    def __init__(self, problem_description: str, evaluation_criteria: str):
        """Initialize prompt sampler."""
        self.problem_description = problem_description
        self.evaluation_criteria = evaluation_criteria
        self.evolution_strategies = [
            self._mutation_prompt,
            self._crossover_prompt,
            self._optimization_prompt,
            self._exploration_prompt,
            self._refinement_prompt
        ]
    
    def generate_initial_prompt(self) -> EvolutionPrompt:
        """Generate prompt for initial population."""
        system_prompt = """You are an expert programmer tasked with solving algorithmic problems.
Generate clean, efficient, and correct Python code that solves the given problem.
Focus on performance and correctness. Always include proper error handling."""
        
        user_prompt = f"""Problem Description:
{self.problem_description}

Evaluation Criteria:
{self.evaluation_criteria}

Please provide a Python solution to this problem. The code should be complete and runnable.
Include any necessary imports at the beginning of the code.

IMPORTANT: Start with a simple working solution. For the circle packing problem, you might start with a simple grid arrangement or random placement that satisfies the constraints, even if it's not optimal.

Example structure:
```python
def pack_circles(n):
    # Your algorithm here
    centers = []  # List of (x, y) tuples
    radius = 0.05  # Start with a reasonable radius
    
    # Generate circle positions
    # ... your code ...
    
    return centers, radius
```"""
        
        return EvolutionPrompt(system_prompt=system_prompt, user_prompt=user_prompt)
    
    def generate_evolution_prompt(self, 
                                parent: Optional[Program] = None,
                                population: List[Program] = None,
                                strategy: Optional[str] = None) -> EvolutionPrompt:
        """Generate prompt for evolving existing programs."""
        if strategy:
            strategy_map = {
                "mutation": self._mutation_prompt,
                "crossover": self._crossover_prompt,
                "optimization": self._optimization_prompt,
                "exploration": self._exploration_prompt,
                "refinement": self._refinement_prompt
            }
            strategy_func = strategy_map.get(strategy, random.choice(self.evolution_strategies))
        else:
            strategy_func = random.choice(self.evolution_strategies)
        
        return strategy_func(parent, population)
    
    def _mutation_prompt(self, parent: Program, population: List[Program]) -> EvolutionPrompt:
        """Generate mutation prompt."""
        system_prompt = """You are an expert programmer specializing in code optimization and evolution.
Your task is to mutate existing code to improve its performance while maintaining correctness."""
        
        user_prompt = f"""Problem Description:
{self.problem_description}

Evaluation Criteria:
{self.evaluation_criteria}

Parent Program (Score: {parent.score:.4f}):
```python
{parent.code}
```

Please mutate this program to improve its performance. Consider:
1. Algorithmic improvements
2. Data structure optimizations
3. Computational efficiency
4. Memory usage
5. Edge case handling

Generate an improved version that maintains correctness while achieving better performance."""
        
        return EvolutionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parent_program=parent
        )
    
    def _crossover_prompt(self, parent: Program, population: List[Program]) -> EvolutionPrompt:
        """Generate crossover prompt combining features from multiple programs."""
        # Select another high-performing program for crossover
        other_programs = [p for p in population if p.id != parent.id and p.score is not None]
        if not other_programs:
            return self._mutation_prompt(parent, population)
        
        other = max(other_programs, key=lambda p: p.score)
        
        system_prompt = """You are an expert programmer specializing in genetic programming.
Your task is to combine the best features from multiple programs to create a superior solution."""
        
        user_prompt = f"""Problem Description:
{self.problem_description}

Evaluation Criteria:
{self.evaluation_criteria}

Program A (Score: {parent.score:.4f}):
```python
{parent.code}
```

Program B (Score: {other.score:.4f}):
```python
{other.code}
```

Analyze both programs and create a new solution that combines their best features.
The new program should leverage the strengths of both approaches while avoiding their weaknesses."""
        
        return EvolutionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parent_program=parent,
            context_programs=[other]
        )
    
    def _optimization_prompt(self, parent: Program, population: List[Program]) -> EvolutionPrompt:
        """Generate prompt focused on specific optimizations."""
        optimization_focus = random.choice([
            "time complexity",
            "space complexity",
            "cache efficiency",
            "vectorization",
            "parallelization",
            "numerical stability"
        ])
        
        system_prompt = f"""You are an expert in {optimization_focus} optimization.
Your task is to optimize existing code with a specific focus on improving {optimization_focus}."""
        
        user_prompt = f"""Problem Description:
{self.problem_description}

Evaluation Criteria:
{self.evaluation_criteria}

Current Program (Score: {parent.score:.4f}):
```python
{parent.code}
```

Optimize this program with a specific focus on {optimization_focus}.
Maintain correctness while achieving significant improvements in {optimization_focus}."""
        
        return EvolutionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parent_program=parent
        )
    
    def _exploration_prompt(self, parent: Program, population: List[Program]) -> EvolutionPrompt:
        """Generate prompt for exploring new algorithmic approaches."""
        system_prompt = """You are a creative algorithm designer.
Your task is to explore novel algorithmic approaches that might not be immediately obvious."""
        
        # Show top performers to avoid
        top_programs = sorted([p for p in population if p.score is not None], 
                            key=lambda p: p.score, reverse=True)[:3]
        
        existing_approaches = "\n\n".join([
            f"Approach {i+1} (Score: {p.score:.4f}):\n```python\n{p.code}\n```"
            for i, p in enumerate(top_programs)
        ])
        
        user_prompt = f"""Problem Description:
{self.problem_description}

Evaluation Criteria:
{self.evaluation_criteria}

Existing approaches:
{existing_approaches}

Generate a completely different algorithmic approach to solve this problem.
Think outside the box and explore unconventional solutions that might achieve breakthrough performance."""
        
        return EvolutionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_programs=top_programs
        )
    
    def _refinement_prompt(self, parent: Program, population: List[Program]) -> EvolutionPrompt:
        """Generate prompt for fine-tuning and polishing."""
        system_prompt = """You are a code optimization expert specializing in fine-tuning.
Your task is to make small but impactful improvements to already high-performing code."""
        
        # Analyze parent's metrics if available
        metrics_analysis = ""
        if parent.metrics:
            metrics_analysis = f"\nPerformance Metrics:\n{parent.metrics}"
        
        user_prompt = f"""Problem Description:
{self.problem_description}

Evaluation Criteria:
{self.evaluation_criteria}

Current Best Program (Score: {parent.score:.4f}):
```python
{parent.code}
```
{metrics_analysis}

This program is already performing well. Make targeted refinements to push its performance even further.
Focus on:
1. Micro-optimizations
2. Constant factor improvements
3. Better handling of edge cases
4. Reducing overhead
5. Improving numerical precision

Every small improvement counts."""
        
        return EvolutionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parent_program=parent
        )