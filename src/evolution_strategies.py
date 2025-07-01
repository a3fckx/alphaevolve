"""Evolution strategies for program selection and population management."""

import random
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from .database import Program


@dataclass
class Island:
    """Represents an island in island-based evolution."""
    id: int
    population: List[Program]
    best_score: float = float('-inf')
    stagnation_counter: int = 0


class EvolutionStrategy:
    """Manages evolution strategies for program selection and population management."""
    
    def __init__(self, 
                 population_size: int = 50,
                 elite_size: int = 5,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.8,
                 crossover_rate: float = 0.2,
                 exploration_rate: float = 0.1):
        """Initialize evolution strategy."""
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.exploration_rate = exploration_rate
        
        # Island model parameters
        self.num_islands = 4
        self.migration_rate = 0.1
        self.migration_interval = 5
    
    def initialize_population(self, programs: List[Program]) -> List[Program]:
        """Initialize population, filling with duplicates if needed."""
        if len(programs) >= self.population_size:
            return programs[:self.population_size]
        
        # Fill remaining slots by duplicating existing programs
        population = programs.copy()
        while len(population) < self.population_size:
            population.append(random.choice(programs))
        
        return population
    
    def select_parent(self, population: List[Program], method: str = "tournament") -> Program:
        """Select a parent for reproduction."""
        if method == "tournament":
            return self._tournament_selection(population)
        elif method == "roulette":
            return self._roulette_selection(population)
        elif method == "rank":
            return self._rank_selection(population)
        else:
            return random.choice(population)
    
    def _tournament_selection(self, population: List[Program]) -> Program:
        """Tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda p: p.score if p.score is not None else float('-inf'))
    
    def _roulette_selection(self, population: List[Program]) -> Program:
        """Roulette wheel selection based on fitness."""
        # Normalize scores to positive values
        scores = [p.score if p.score is not None else 0 for p in population]
        min_score = min(scores)
        normalized_scores = [s - min_score + 1 for s in scores]
        
        total = sum(normalized_scores)
        if total == 0:
            return random.choice(population)
        
        r = random.uniform(0, total)
        cumsum = 0
        for program, score in zip(population, normalized_scores):
            cumsum += score
            if cumsum >= r:
                return program
        
        return population[-1]
    
    def _rank_selection(self, population: List[Program]) -> Program:
        """Rank-based selection."""
        sorted_pop = sorted(population, 
                          key=lambda p: p.score if p.score is not None else float('-inf'))
        
        # Assign ranks (worst=1, best=n)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        
        r = random.uniform(0, total_rank)
        cumsum = 0
        for program, rank in zip(sorted_pop, ranks):
            cumsum += rank
            if cumsum >= r:
                return program
        
        return sorted_pop[-1]
    
    def select_evolution_strategy(self) -> str:
        """Select which evolution strategy to use with adaptive weighting."""
        # Adjust rates dynamically based on generation and diversity
        diversity = self.diversity_score(self.population) if hasattr(self, 'population') else 0.5
        generation_factor = min(1.0, max(0.0, self.generation / 50.0) if hasattr(self, 'generation') else 0.0)
        
        # Increase diff_mutation in later generations for fine-tuning
        adjusted_diff_rate = 0.1 + generation_factor * 0.3
        adjusted_mutation_rate = self.mutation_rate * (1.0 - diversity * 0.2)
        adjusted_crossover_rate = self.crossover_rate * (1.0 + diversity * 0.1)
        adjusted_exploration_rate = self.exploration_rate * (1.0 + diversity * 0.3)
        
        total = adjusted_mutation_rate + adjusted_crossover_rate + adjusted_exploration_rate + adjusted_diff_rate
        r = random.uniform(0, total)
        
        if r < adjusted_mutation_rate:
            return "mutation"
        elif r < adjusted_mutation_rate + adjusted_crossover_rate:
            return "crossover"
        elif r < adjusted_mutation_rate + adjusted_crossover_rate + adjusted_exploration_rate:
            return "exploration"
        else:
            return "diff_mutation"  # Enhanced diff-based mutation for incremental improvements
    
    def select_elite(self, population: List[Program]) -> List[Program]:
        """Select elite programs to preserve."""
        sorted_pop = sorted(population, 
                          key=lambda p: p.score if p.score is not None else float('-inf'), 
                          reverse=True)
        return sorted_pop[:self.elite_size]
    
    def create_islands(self, population: List[Program]) -> List[Island]:
        """Divide population into islands for island-based evolution."""
        islands = []
        island_size = len(population) // self.num_islands
        
        for i in range(self.num_islands):
            start = i * island_size
            end = start + island_size if i < self.num_islands - 1 else len(population)
            island_pop = population[start:end]
            
            islands.append(Island(
                id=i,
                population=island_pop,
                best_score=max((p.score for p in island_pop if p.score is not None), 
                             default=float('-inf'))
            ))
        
        return islands
    
    def migrate_between_islands(self, islands: List[Island]) -> None:
        """Perform migration between islands."""
        for island in islands:
            if random.random() < self.migration_rate:
                # Select best individual from island
                best = max(island.population, 
                         key=lambda p: p.score if p.score is not None else float('-inf'))
                
                # Migrate to random other island
                target_island = random.choice([i for i in islands if i.id != island.id])
                target_island.population.append(best)
                
                # Remove worst from target island to maintain size
                if len(target_island.population) > len(island.population):
                    worst = min(target_island.population,
                              key=lambda p: p.score if p.score is not None else float('-inf'))
                    target_island.population.remove(worst)
    
    def update_island_stats(self, island: Island) -> None:
        """Update island statistics and detect stagnation."""
        current_best = max((p.score for p in island.population if p.score is not None), 
                         default=float('-inf'))
        
        if current_best > island.best_score:
            island.best_score = current_best
            island.stagnation_counter = 0
        else:
            island.stagnation_counter += 1
    
    def handle_stagnation(self, island: Island) -> None:
        """Handle stagnation in an island."""
        if island.stagnation_counter > 10:
            # Keep only top 20% and regenerate the rest
            sorted_pop = sorted(island.population,
                              key=lambda p: p.score if p.score is not None else float('-inf'),
                              reverse=True)
            keep_size = max(1, len(island.population) // 5)
            island.population = sorted_pop[:keep_size]
            island.stagnation_counter = 0
    
    def diversity_score(self, population: List[Program]) -> float:
        """Calculate diversity score of population based on code similarity."""
        if len(population) < 2:
            return 1.0
        
        # Enhanced diversity based on code content variation using simple token comparison
        def code_similarity(code1: str, code2: str) -> float:
            tokens1 = set(code1.split())
            tokens2 = set(code2.split())
            if not tokens1 or not tokens2:
                return 0.0
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return intersection / union if union > 0 else 0.0
        
        total_similarity = 0.0
        comparisons = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_similarity += code_similarity(population[i].code, population[j].code)
                comparisons += 1
        
        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
        # Diversity is inverse of average similarity
        return 1.0 - avg_similarity
    
    def adaptive_parameters(self, generation: int, diversity: float) -> None:
        """Adapt evolution parameters based on progress and diversity."""
        self.generation = generation  # Store for use in strategy selection
        # Increase exploration if diversity is low
        if diversity < 0.2:
            self.exploration_rate = min(0.4, self.exploration_rate * 1.2)
            self.mutation_rate = min(0.9, self.mutation_rate * 1.1)
        else:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        
        # Increase exploitation and diff_mutation focus in later generations
        if generation > 50:
            self.crossover_rate = min(0.5, self.crossover_rate * 1.05)
            self.mutation_rate = max(0.3, self.mutation_rate * 0.9)
        else:
            self.crossover_rate = max(0.1, self.crossover_rate * 0.95)
