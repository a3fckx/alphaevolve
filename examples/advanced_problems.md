# Advanced Problems for AlphaEvolve

This document discusses advanced problems implemented in AlphaEvolve, highlighting the examples we are currently evaluating, the current state-of-the-art (SOTA), and notable achievements from the AlphaEvolve paper by Google DeepMind.

## 1. Circle Packing
- **Problem**: Pack a specified number of equal circles (e.g., 26) into a unit square to maximize the radius while ensuring no overlaps and all circles are within the square boundaries.
- **Challenge**: This is a classic optimization problem with applications in logistics and materials science, requiring efficient spatial arrangement.
- **Current SOTA**: Known optimal packings exist for specific numbers of circles; for 26 circles, the best-known radius is approximately 0.0979.
- **AlphaEvolve Achievement**: As detailed in the AlphaEvolve technical report (Novikov et al., 2025), AlphaEvolve approached near-optimal solutions for circle packing, often matching or closely approximating SOTA radii through evolutionary code generation. The [mathematical_results.ipynb](https://github.com/google-deepmind/alphaevolve_results/blob/main/mathematical_results.ipynb) notebook in the AlphaEvolve Results Repository provides specific instances where AlphaEvolve outperformed previous best-known packings for certain numbers of circles.

## 2. Function Optimization
- **Problem**: Minimize complex multi-dimensional functions, such as the 10-dimensional Rastrigin function, which is a benchmark for optimization algorithms.
- **Challenge**: The problem tests the ability to navigate high-dimensional search spaces with many local minima to find the global minimum.
- **Current SOTA**: Advanced optimization algorithms like Differential Evolution and Particle Swarm Optimization achieve high performance on benchmark functions like Rastrigin, with known global minima (0 for Rastrigin).
- **AlphaEvolve Achievement**: According to the AlphaEvolve technical report (Novikov et al., 2025), AlphaEvolve has shown capability in generating algorithms that effectively explore high-dimensional spaces, achieving results competitive with traditional optimization methods for such functions. Detailed outcomes can be explored in the [AlphaEvolve Results Repository](https://github.com/google-deepmind/alphaevolve_results).

## 3. Kissing Spheres (Kissing Number Problem)
- **Problem**: Find the maximum number of non-overlapping unit spheres that can touch a central unit sphere in n dimensions, or optimize sphere arrangements to maximize kissing points.
- **Challenge**: A famous problem originating from Isaac Newton, with applications in materials science, cryptography, and coding theory.
- **Current SOTA**: Known results include K(3) = 12, K(4) = 24, K(8) = 240, and K(24) = 196,560, with proofs for some dimensions and conjectures for others.
- **AlphaEvolve Achievement**: The AlphaEvolve technical report (Novikov et al., 2025) highlights exploration of geometric optimization problems like this, demonstrating the potential to rediscover known configurations and propose novel arrangements in higher dimensions. The [mathematical_results.ipynb](https://github.com/google-deepmind/alphaevolve_results/blob/main/mathematical_results.ipynb) notebook in the AlphaEvolve Results Repository details specific dimensions where AlphaEvolve achieved improvements over previously known configurations.

## 4. Matrix Multiplication Optimization
- **Problem**: Develop algorithms to multiply nxn matrices using fewer scalar multiplications than the standard O(n³) approach, optimizing for speed with large matrices.
- **Challenge**: Reducing the number of operations or optimizing memory access patterns can significantly impact performance, especially for hardware-specific implementations.
- **Current SOTA**: Strassen's algorithm (1969) reduced the number of multiplications for 4x4 matrices to 49; subsequent research has further optimized specific cases, with theoretical lower bounds still under investigation.
- **AlphaEvolve Achievement**: As reported in the AlphaEvolve technical report (Novikov et al., 2025), the system discovered a 48-multiplication solution for 4x4 matrices, improving upon Strassen's algorithm, showcasing its ability to find novel algorithmic efficiencies. Further details and verification code are available in the [AlphaEvolve Results Repository](https://github.com/google-deepmind/alphaevolve_results).

## Implementation Notes

For each implemented problem, we have ensured:
1. **Clear Mathematical Formulation**: Each problem is precisely defined with constraints and objectives.
2. **Automated Evaluator**: Custom evaluators are implemented to score solutions based on multi-objective criteria (e.g., radius for circle packing, execution time for matrix multiplication).
3. **Benchmark Datasets**: Where applicable, known best solutions or reference implementations are used for comparison.
4. **Performance Metrics**: Metrics include solution quality (e.g., achieved radius, minimized function value), correctness, and execution efficiency.

## Evolution Strategy Comparison

Our current implementation of AlphaEvolve employs a combination of mutation, crossover, and exploration strategies to evolve programs, focusing on generating new variants through these methods. We use mechanisms like tournament selection, elite preservation, and an island model to maintain population diversity and drive improvement. However, this differs from the original AlphaEvolve approach as described in the technical report (Novikov et al., 2025), which primarily worked with diffs and small incremental improvements to existing code. In the original methodology, problem skeletons were often provided, and the algorithm evolved specific parts of the code to solve the problem, focusing on targeted modifications rather than full code regeneration.

In our implementation, we currently generate complete solutions without explicitly using diffs or predefined skeletons for most problems. This approach allows for broader exploration but may lack the fine-grained control and efficiency of evolving code through diffs. Future improvements could include integrating a diff-based evolution strategy and providing problem-specific skeletons to align more closely with the original AlphaEvolve methodology, potentially enhancing the precision and effectiveness of the evolutionary process.

The key characteristics of these advanced problems:
- They often have decades of research behind them, making any improvement significant.
- Small performance gains (even 0.001%) can have substantial real-world impact.
- Solutions may require deep mathematical or algorithmic insights, which AlphaEvolve aims to uncover through evolutionary processes.
- Evaluation can be computationally intensive, reflecting the complexity of real-world optimization challenges.
