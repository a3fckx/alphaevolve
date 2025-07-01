# AlphaEvolve: Evolutionary Strategies

This document provides a detailed explanation of the evolutionary strategies implemented in `src/evolution_strategies.py`. This component is the core engine driving the "evolution" in AlphaEvolve, managing how populations of programs are selected, reproduced, and diversified over generations.

## Core Concepts

Before diving into the details, it's important to understand a few core concepts of evolutionary algorithms:

-   **Population:** A collection of candidate programs (potential solutions) for the given algorithmic problem.
-   **Generation:** A single iteration of the evolutionary loop. In each generation, a new population is created from the previous one, hopefully containing better solutions.
-   **Fitness:** A numerical score that quantifies how well a program solves the problem. In AlphaEvolve, this is the `score` attribute of a `Program` object, determined by the `Evaluator`.
-   **Evolution Strategies:** The set of rules and methods used to create the next generation of programs from the current one. The goal is to balance "exploitation" (improving existing good solutions) and "exploration" (searching for new, different solutions).

---

## The `EvolutionStrategy` Class

This is the central class that orchestrates the entire evolutionary process.

### Initialization (`__init__`)

When an `EvolutionStrategy` object is created, it's configured with several key parameters that control the behavior of the evolution:

-   `population_size`: The total number of programs to maintain in each generation.
-   `elite_size`: The number of the absolute best-performing programs that are automatically carried over to the next generation, unchanged. This is a concept called **Elitism**, and it ensures that the best solutions found so far are never lost.
-   `tournament_size`: A parameter for Tournament Selection, defining how many programs compete to be chosen as a parent.
-   `mutation_rate`: The probability that a new program will be created by "mutating" a single parent.
-   `crossover_rate`: The probability that a new program will be created by "crossing over" (combining) two parent programs.
-   `exploration_rate`: The probability that a new program will be generated from scratch, without any parents, to introduce novel ideas into the population.

### Parent Selection (`select_parent`)

To create a new program via mutation or crossover, one or two "parents" must be selected from the current population. The selection is probabilistic, giving fitter programs a higher chance of being chosen. This file implements three standard selection methods:

1.  **Tournament Selection (Default):** A small, random subset of the population (of size `tournament_size`) is chosen. These programs then "compete," and the one with the highest fitness score is selected as the parent. This is efficient and effective at selecting good candidates without giving an overwhelming advantage to the single best program.
2.  **Roulette Wheel Selection:** Each program is assigned a portion of a "roulette wheel" that is proportional to its fitness score. A random spin of the wheel determines the parent. This method gives every program a chance to be selected, but higher-scoring programs have a better chance.
3.  **Rank Selection:** This method is similar to the roulette wheel, but it uses the program's rank (1st, 2nd, 3rd, etc.) instead of its raw score. This prevents a single program with a massive score from dominating the selection process and improves population diversity.

### Creating New Programs (`select_evolution_strategy`)

Once a parent (or parents) is selected, this method decides *how* to create the new program. It randomly chooses one of the following strategies based on the rates set during initialization:

-   **Mutation (High Probability):** This is the primary driver of evolution. It takes a single parent program and prompts the LLM to make a targeted change or improvement. This is analogous to a genetic mutation.
-   **Crossover (Medium Probability):** This strategy takes two parent programs and prompts the LLM to combine their best features into a new "child" program. This is analogous to sexual reproduction and can lead to powerful new combinations of ideas.
-   **Exploration (Low Probability):** This strategy ignores the current population and prompts the LLM to generate a completely new solution from scratch. This is vital for introducing genetic diversity and preventing the population from getting stuck on a suboptimal solution (a "local optimum").
-   **Refinement (Fallback):** If none of the above are chosen, the system defaults to a "refinement" strategy, which likely involves asking the LLM to simply improve upon an existing solution without a specific mutation or crossover instruction.

---

## Advanced Concepts

The implementation in `evolution_strategies.py` includes several advanced techniques that make the evolutionary process more robust and effective.

### 1. Island Model

-   **What it is:** Instead of having one large, interbreeding population, the Island Model divides the population into several smaller, isolated sub-populations called "islands."
-   **Why it's used:** The primary goal is to **maintain genetic diversity**. In a single large population, a few good solutions can quickly come to dominate, causing the algorithm to converge prematurely on a suboptimal solution. By evolving populations in isolation, each island can explore a different "region" of the solution space.
-   **How it's implemented:**
    1.  **Creation (`create_islands`):** The initial population is split evenly among a fixed number of islands.
    2.  **Isolation:** For most generations, the islands evolve independently. They have their own populations, and parent selection only occurs within the island.
    3.  **Migration (`migrate_between_islands`):** Periodically (e.g., every 5 generations), a "migration" event occurs. The best program from one island is copied and moved to a randomly chosen different island. This allows successful "genes" to spread across the entire population, while still allowing for independent evolution most of the time. The receiving island removes its worst member to maintain its population size.

### 2. Stagnation Handling

-   **What it is:** Stagnation occurs when an island's best fitness score stops improving for several consecutive generations. The island is "stuck."
-   **Why it's used:** This is a mechanism to jolt a stuck population back into productive evolution.
-   **How it's implemented (`handle_stagnation`):** The system tracks the number of generations an island has gone without improving its best score. If this "stagnation counter" exceeds a threshold (e.g., 10 generations), a drastic event is triggered: the majority of the island's population is culled, keeping only the top few elite programs. The empty slots are then filled with new programs, forcing a fresh start and hopefully breaking out of the local optimum.

### 3. Diversity Score

-   **What it is:** A metric to quantify how different the programs in a population are from one another.
-   **Why it's used:** Monitoring diversity is crucial. If diversity is low, it's a sign that the population is converging and the risk of getting stuck is high.
-   **How it's implemented (`diversity_score`):** The current implementation uses a simple but effective proxy for diversity: it calculates the statistical variance of the *length* of the code in each program. A high variance means there is a mix of long and short programs, suggesting more structural diversity. A low variance suggests the programs are all becoming very similar.

### 4. Adaptive Parameters

-   **What it is:** This is perhaps the most sophisticated feature. The system can dynamically adjust its own evolutionary parameters in response to the state of the evolution.
-   **Why it's used:** A fixed set of parameters may not be optimal for the entire evolutionary run. The ideal strategy might change over time.
-   **How it's implemented (`adaptive_parameters`):**
    -   If the **diversity score is low**, it's a sign that more exploration is needed. The system will automatically increase the `exploration_rate` and `mutation_rate` to encourage more novel and varied solutions.
    -   In **later generations**, the system might assume it's closer to a good solution. It will then slightly decrease the `exploration_rate` and increase the `crossover_rate` to focus more on "exploiting" and combining the good solutions it has already found, rather than searching for brand new ones.
