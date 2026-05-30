"""
Sholomon et al. 2013 Genetic Algorithm-Based Jigsaw Puzzle Solver

Reference:
Dror Sholomon, Omid David, and Nathan S Netanyahu.
A genetic algorithm-based solver for very large jigsaw puzzles.
In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 1767–1774, 2013.

Note: This implementation assumes Type 1 puzzles where all pieces have 
the correct orientation (no rotation). This is the standard assumption 
for most jigsaw puzzle solvers and matches your GAP dataset format.

For Type 1 puzzles:
- Each piece has a fixed orientation
- Only position (permutation) needs to be solved
- Chromosome is a permutation of piece indices
- Much simpler search space than Type 2 puzzles with unknown rotations
"""

import numpy as np
from typing import List, Tuple, Optional
import time

# Constants for edge compatibility
P = 0.3
Q = 0.0625

# Edge directions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


def calculate_dissimilarity(x_i: np.ndarray, x_j: np.ndarray, relation: int) -> float:
    """
    Calculate dissimilarity between two puzzle pieces along a given edge.
    Uses the same metric as Pomeranz et al.
    
    Args:
        x_i: First piece (H, W, C) in LAB color space
        x_j: Second piece (H, W, C) in LAB color space
        relation: Edge direction (LEFT, RIGHT, UP, DOWN)
    
    Returns:
        Dissimilarity score (lower is better)
    """
    nrows, ncols = x_i.shape[0], x_i.shape[1]

    if relation == LEFT:
        return calculate_dissimilarity(x_j, x_i, RIGHT)
    elif relation == RIGHT:
        # Compare right edge of x_i with left edge of x_j
        return np.sum(
            np.power(
                np.power(np.abs((2 * x_i[:, ncols - 1] - x_i[:, ncols - 2]) - x_j[:, 0]), P) +
                np.power(np.abs((2 * x_j[:, 0] - x_j[:, 1]) - x_i[:, ncols - 1]), P), 
                Q / P
            )
        )
    elif relation == UP:
        return calculate_dissimilarity(x_j, x_i, DOWN)
    elif relation == DOWN:
        # Compare bottom edge of x_i with top edge of x_j
        return np.sum(
            np.power(
                np.power(np.abs((2 * x_i[nrows - 1] - x_i[nrows - 2]) - x_j[0]), P) +
                np.power(np.abs((2 * x_j[0] - x_j[1]) - x_i[nrows - 1]), P), 
                Q / P
            )
        )
    else:
        raise ValueError(f'invalid relation: {relation}')


def build_dissimilarity_matrix(pieces: List[np.ndarray]) -> np.ndarray:
    """
    Build pairwise dissimilarity matrix for all pieces.
    
    Args:
        pieces: List of puzzle pieces in LAB color space
    
    Returns:
        Dissimilarity matrix of shape (4, N, N) for 4 directions
    """
    n_pieces = len(pieces)
    dissimilarity_matrix = np.zeros((4, n_pieces, n_pieces))
    
    for i in range(n_pieces):
        for j in range(n_pieces):
            if i == j:
                continue
            for relation in range(4):
                dissimilarity_matrix[relation][i][j] = calculate_dissimilarity(
                    pieces[i], pieces[j], relation
                )
    
    return dissimilarity_matrix


def calculate_fitness(individual: np.ndarray, dissimilarity_matrix: np.ndarray, 
                     grid_size: int) -> float:
    """
    Calculate fitness of an individual (lower dissimilarity = higher fitness).
    Fitness is the negative sum of dissimilarities between adjacent pieces.
    
    Args:
        individual: 1D array of piece indices (permutation)
        dissimilarity_matrix: Precomputed dissimilarity matrix (4, N, N)
        grid_size: Size of puzzle grid (e.g., 3 for 3x3)
    
    Returns:
        Fitness score (higher is better)
    """
    total_dissimilarity = 0.0
    
    # Reshape to 2D grid
    grid = individual.reshape(grid_size, grid_size)
    
    # Sum dissimilarities for all adjacent pairs
    for i in range(grid_size):
        for j in range(grid_size):
            piece_id = grid[i, j]
            
            # Right neighbor
            if j < grid_size - 1:
                neighbor_id = grid[i, j + 1]
                total_dissimilarity += dissimilarity_matrix[RIGHT][piece_id][neighbor_id]
            
            # Bottom neighbor
            if i < grid_size - 1:
                neighbor_id = grid[i + 1, j]
                total_dissimilarity += dissimilarity_matrix[DOWN][piece_id][neighbor_id]
    
    # Fitness is negative dissimilarity (we want to minimize dissimilarity)
    return -total_dissimilarity


def create_initial_population(pop_size: int, n_pieces: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Create initial population of random permutations.
    
    Args:
        pop_size: Population size
        n_pieces: Number of puzzle pieces
        seed: Random seed for reproducibility
    
    Returns:
        Population array of shape (pop_size, n_pieces)
    """
    if seed is not None:
        np.random.seed(seed)
    
    population = np.zeros((pop_size, n_pieces), dtype=np.int64)
    for i in range(pop_size):
        population[i] = np.random.permutation(n_pieces)
    
    return population


def pmx_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partially Mapped Crossover (PMX) for permutation chromosomes.
    
    Args:
        parent1: First parent permutation
        parent2: Second parent permutation
    
    Returns:
        Two offspring permutations
    """
    size = len(parent1)
    
    # Select two crossover points
    cx_point1, cx_point2 = sorted(np.random.choice(range(size), 2, replace=False))
    
    # Initialize offspring as copies of parents
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    # Create mapping between middle sections
    mapping1 = {}
    mapping2 = {}
    
    for i in range(cx_point1, cx_point2 + 1):
        mapping1[parent2[i]] = parent1[i]
        mapping2[parent1[i]] = parent2[i]
        offspring1[i] = parent2[i]
        offspring2[i] = parent1[i]
    
    # Fix conflicts outside the middle section
    for i in list(range(cx_point1)) + list(range(cx_point2 + 1, size)):
        # Fix offspring1
        while offspring1[i] in offspring1[cx_point1:cx_point2 + 1]:
            offspring1[i] = mapping1[offspring1[i]]
        
        # Fix offspring2
        while offspring2[i] in offspring2[cx_point1:cx_point2 + 1]:
            offspring2[i] = mapping2[offspring2[i]]
    
    return offspring1, offspring2


def swap_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Swap mutation: randomly swap two positions.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
    
    Returns:
        Mutated individual
    """
    if np.random.random() < mutation_rate:
        mutated = individual.copy()
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    return individual


def inversion_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Inversion mutation: reverse a random subsequence.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
    
    Returns:
        Mutated individual
    """
    if np.random.random() < mutation_rate:
        mutated = individual.copy()
        idx1, idx2 = sorted(np.random.choice(len(individual), 2, replace=False))
        mutated[idx1:idx2 + 1] = mutated[idx1:idx2 + 1][::-1]
        return mutated
    return individual


def scramble_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Scramble mutation: randomly shuffle a subsequence.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
    
    Returns:
        Mutated individual
    """
    if np.random.random() < mutation_rate:
        mutated = individual.copy()
        idx1, idx2 = sorted(np.random.choice(len(individual), 2, replace=False))
        subset = mutated[idx1:idx2 + 1].copy()
        np.random.shuffle(subset)
        mutated[idx1:idx2 + 1] = subset
        return mutated
    return individual


def tournament_selection(population: np.ndarray, fitness_scores: np.ndarray, 
                        tournament_size: int = 3) -> np.ndarray:
    """
    Tournament selection: select best individual from random tournament.
    
    Args:
        population: Current population
        fitness_scores: Fitness of each individual
        tournament_size: Number of individuals in tournament
    
    Returns:
        Selected individual
    """
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = fitness_scores[tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx].copy()


def elitism_selection(population: np.ndarray, fitness_scores: np.ndarray, 
                     n_elite: int) -> np.ndarray:
    """
    Select top n individuals based on fitness.
    
    Args:
        population: Current population
        fitness_scores: Fitness of each individual
        n_elite: Number of elite individuals to select
    
    Returns:
        Elite individuals
    """
    elite_indices = np.argsort(fitness_scores)[-n_elite:]
    return population[elite_indices].copy()


def genetic_algorithm_solver(pieces: List[np.ndarray], 
                            grid_size: int,
                            pop_size: int = 100,
                            n_generations: int = 1000,
                            mutation_rate: float = 0.01,
                            crossover_rate: float = 0.8,
                            elitism_ratio: float = 0.1,
                            tournament_size: int = 3,
                            early_stopping: int = 100,
                            seed: Optional[int] = None,
                            verbose: bool = False) -> Tuple[np.ndarray, float, dict]:
    """
    Solve jigsaw puzzle using genetic algorithm.
    
    Args:
        pieces: List of puzzle pieces in LAB color space
        grid_size: Grid size (e.g., 3 for 3x3 puzzle)
        pop_size: Population size
        n_generations: Maximum number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        elitism_ratio: Fraction of population to keep as elite
        tournament_size: Size of tournament for selection
        early_stopping: Stop if no improvement for this many generations
        seed: Random seed
        verbose: Print progress
    
    Returns:
        best_individual: Best solution found (1D permutation)
        best_fitness: Fitness of best solution
        stats: Dictionary with algorithm statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_pieces = len(pieces)
    n_elite = max(1, int(pop_size * elitism_ratio))
    
    # Build dissimilarity matrix
    if verbose:
        print("Building dissimilarity matrix...")
    dissimilarity_matrix = build_dissimilarity_matrix(pieces)
    
    # Initialize population
    if verbose:
        print(f"Initializing population (size={pop_size})...")
    population = create_initial_population(pop_size, n_pieces, seed)
    
    # Track best solution
    best_individual = None
    best_fitness = -np.inf
    generations_without_improvement = 0
    
    # Statistics
    fitness_history = []
    
    # Evolution loop
    for generation in range(n_generations):
        # Evaluate fitness
        fitness_scores = np.array([
            calculate_fitness(ind, dissimilarity_matrix, grid_size) 
            for ind in population
        ])
        
        # Track best
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx].copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        fitness_history.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'mean_fitness': fitness_scores.mean(),
            'std_fitness': fitness_scores.std()
        })
        
        if verbose and generation % 50 == 0:
            print(f"Gen {generation}: Best={best_fitness:.2f}, "
                  f"Mean={fitness_scores.mean():.2f}, "
                  f"No improvement={generations_without_improvement}")
        
        # Early stopping
        if generations_without_improvement >= early_stopping:
            if verbose:
                print(f"Early stopping at generation {generation}")
            break
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individuals
        elite = elitism_selection(population, fitness_scores, n_elite)
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitness_scores, tournament_size)
            parent2 = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            if np.random.random() < crossover_rate:
                offspring1, offspring2 = pmx_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation (try different mutation operators)
            mutation_type = np.random.choice(['swap', 'inversion', 'scramble'])
            if mutation_type == 'swap':
                offspring1 = swap_mutation(offspring1, mutation_rate)
                offspring2 = swap_mutation(offspring2, mutation_rate)
            elif mutation_type == 'inversion':
                offspring1 = inversion_mutation(offspring1, mutation_rate)
                offspring2 = inversion_mutation(offspring2, mutation_rate)
            else:
                offspring1 = scramble_mutation(offspring1, mutation_rate)
                offspring2 = scramble_mutation(offspring2, mutation_rate)
            
            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)
        
        population = np.array(new_population[:pop_size])
    
    stats = {
        'n_generations': generation + 1,
        'fitness_history': fitness_history,
        'final_best_fitness': best_fitness,
        'converged': generations_without_improvement >= early_stopping
    }
    
    return best_individual, best_fitness, stats


def solve_puzzle(pieces: List[np.ndarray], 
                grid_size: int,
                pop_size: int = 100,
                n_generations: int = 1000,
                mutation_rate: float = 0.01,
                seed: Optional[int] = None) -> Tuple[np.ndarray, float, dict]:
    """
    Convenience wrapper for genetic_algorithm_solver.
    
    Args:
        pieces: List of puzzle pieces in LAB color space
        grid_size: Grid size (e.g., 3 for 3x3)
        pop_size: Population size
        n_generations: Maximum generations
        mutation_rate: Mutation probability
        seed: Random seed
    
    Returns:
        solution: 2D array of piece indices
        fitness: Fitness score
        stats: Algorithm statistics
    """
    solution, fitness, stats = genetic_algorithm_solver(
        pieces, grid_size,
        pop_size=pop_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        seed=seed,
        verbose=False
    )
    
    # Reshape to 2D grid
    solution_2d = solution.reshape(grid_size, grid_size)
    
    return solution_2d, fitness, stats
