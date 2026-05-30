"""
Pomeranz et al. 2011 Greedy Jigsaw Puzzle Solver
Adapted from: https://github.com/yi-jiayu/shuffled-images

Reference:
Dolev Pomeranz, Michal Shemesh, and Ohad Ben-Shahar. 
A fully automated greedy square jigsaw puzzle solver. 
In CVPR 2011, pages 9–16. IEEE, 2011.

Note: This implementation assumes Type 1 puzzles where all pieces have 
the correct orientation (no rotation). This is the standard assumption 
for most jigsaw puzzle solvers and matches your GAP dataset format.
"""

import numpy as np
import itertools
from typing import Tuple, List

# Constants for edge directions
P = 0.3
Q = 0.0625

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


def calculate_dissimilarity(x_i: np.ndarray, x_j: np.ndarray, relation: int) -> float:
    """
    Calculate dissimilarity between two puzzle pieces along a given edge.
    
    Args:
        x_i: First piece (H, W, C) in LAB color space
        x_j: Second piece (H, W, C) in LAB color space
        relation: Edge direction (LEFT, RIGHT, UP, DOWN)
    
    Returns:
        Dissimilarity score
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


def build_dissimilarity_matrix(squares: List[np.ndarray]) -> np.ndarray:
    """
    Build dissimilarity matrix for all piece pairs and all relations.
    
    Args:
        squares: List of puzzle pieces, each (H, W, C)
    
    Returns:
        Dissimilarity matrix of shape (4, N, N)
    """
    dissimilarity_matrix = np.empty((4, len(squares), len(squares)))
    for i, x_i in enumerate(squares):
        for j, x_j in enumerate(squares):
            for relation in range(4):
                if i == j:
                    continue
                elif i < j:
                    dissimilarity_matrix[relation][i][j] = calculate_dissimilarity(x_i, x_j, relation)
                else:
                    dissimilarity_matrix[relation][i][j] = dissimilarity_matrix[opposite_relation(relation)][j][i]
    return dissimilarity_matrix


def calculate_compatibility(dissimilarity_matrix: np.ndarray, 
                           percentiles: np.ndarray, 
                           i: int, j: int, relation: int) -> float:
    """Calculate compatibility score using normalized exponential."""
    percentile = percentiles[relation][i]
    if percentile == 0:
        percentile = 2.220446049250313e-16
    return np.exp(-dissimilarity_matrix[relation][i][j] / percentile)


def build_compatibility_matrix(squares: List[np.ndarray]) -> np.ndarray:
    """
    Build compatibility matrix from dissimilarity matrix.
    
    Args:
        squares: List of puzzle pieces
    
    Returns:
        Compatibility matrix of shape (4, N, N)
    """
    dissimilarity_matrix = build_dissimilarity_matrix(squares)
    _, order, _ = dissimilarity_matrix.shape

    # Precalculate percentiles
    percentiles = np.empty((4, order))
    for i in range(order):
        for relation in range(4):
            percentiles[relation][i] = np.percentile(np.delete(dissimilarity_matrix[relation][i], i), 25)

    compatibility_matrix = np.empty((4, order, order))
    for i in range(order):
        for j in range(order):
            for relation in range(4):
                if i == j:
                    continue
                elif i < j:
                    compatibility_matrix[relation][i][j] = calculate_compatibility(
                        dissimilarity_matrix, percentiles, i, j, relation
                    )
                else:
                    compatibility_matrix[relation][i][j] = compatibility_matrix[opposite_relation(relation)][j][i]
    return compatibility_matrix


def calculate_best_neighbours(compatibility_matrix: np.ndarray) -> np.ndarray:
    """
    Find best neighbor for each piece in each direction.
    
    Returns:
        Best neighbors array of shape (4, N)
    """
    _, order, _ = compatibility_matrix.shape
    best_neighbours = np.zeros((4, order), dtype=int)
    for relation in range(4):
        for i in range(order):
            best_neighbours[relation][i] = np.argmax(compatibility_matrix[relation][i])
    return best_neighbours


def opposite_relation(relation: int) -> int:
    """Get opposite direction."""
    if relation == 0 or relation == 2:
        return relation + 1
    else:
        return relation - 1


def find_best_estimated_seed(best_neighbours: np.ndarray) -> int:
    """
    Find the piece with most mutual best buddies as starting seed.
    
    Returns:
        Index of best seed piece
    """
    _, order = best_neighbours.shape
    num_best_buddies = np.zeros(order, dtype=int)
    for relation in range(4):
        for i in range(order):
            buddy = best_neighbours[relation][i]
            opposite = opposite_relation(relation)
            if best_neighbours[opposite][buddy] == i:
                num_best_buddies[i] += 1
    return np.argmax(num_best_buddies)


def adjacent(i: int, j: int) -> List[Tuple[int, int]]:
    """Get adjacent positions."""
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


def is_in_grid(grid: np.ndarray, i: int, j: int) -> bool:
    """Check if position is within grid bounds."""
    nrows, ncols = grid.shape
    return 0 <= i < nrows and 0 <= j < ncols


def is_occupied_slot(puzzle: np.ndarray, i: int, j: int) -> bool:
    """Check if a slot is occupied."""
    return is_in_grid(puzzle, i, j) and puzzle[i][j] >= 0


def find_candidate_slots(puzzle: np.ndarray) -> set:
    """
    Find candidate slots adjacent to placed pieces.
    Returns slots with maximum number of occupied neighbors.
    """
    slots = {}
    for i, j in np.argwhere(puzzle != -1):
        for x, y in adjacent(i, j):
            if (x, y) in slots:
                continue
            if is_occupied_slot(puzzle, x, y):
                continue
            # Count occupied neighbors
            slots[(x, y)] = sum(1 if is_occupied_slot(puzzle, p, q) else 0 for p, q in adjacent(x, y))
    
    if not slots:
        return set()
    
    max_neighbours = max(slots.values())
    return set(slot for slot, num_neighbours in slots.items() if num_neighbours == max_neighbours)


def best_buddies(best_neighbours: np.ndarray, relation: int, i: int, j: int) -> bool:
    """Check if two pieces are mutual best buddies."""
    return best_neighbours[relation][i] == j and best_neighbours[opposite_relation(relation)][j] == i


def does_part_fit_in_slot(puzzle: np.ndarray, best_neighbours: np.ndarray, 
                         slot: Tuple[int, int], part: int) -> bool:
    """
    Check if a part fits in a slot (must be best buddies with all occupied neighbors).
    """
    i, j = slot
    for relation in range(4):
        x, y = related_coords(relation, i, j)
        if is_occupied_slot(puzzle, x, y):
            if not best_buddies(best_neighbours, relation, part, puzzle[x][y]):
                return False
    return True


def related_coords(relation: int, i: int, j: int) -> Tuple[int, int]:
    """Get coordinates of neighbor in given direction."""
    if relation == RIGHT:
        return i, j + 1
    elif relation == LEFT:
        return i, j - 1
    elif relation == UP:
        return i - 1, j
    elif relation == DOWN:
        return i + 1, j
    else:
        raise ValueError(f'invalid relation: {relation}')


def average_compatibility_with_slot(puzzle: np.ndarray, compatibility_matrix: np.ndarray, 
                                   slot: Tuple[int, int], part: int) -> float:
    """Calculate average compatibility of a part with occupied neighbors of a slot."""
    i, j = slot
    total_compatibility = 0
    num_neighbours = 0

    for relation in range(4):
        x, y = related_coords(relation, i, j)
        if is_occupied_slot(puzzle, x, y):
            total_compatibility += compatibility_matrix[relation][part][puzzle[x][y]]
            num_neighbours += 1

    return total_compatibility / num_neighbours if num_neighbours > 0 else 0


class SlotAssignError(Exception):
    """Exception raised when a slot assignment is invalid."""
    pass


def try_assign(puzzle: np.ndarray, slot: Tuple[int, int], 
              part: int, unallocated_parts: set) -> Tuple[np.ndarray, set]:
    """
    Try to assign a part to a slot, rolling puzzle if needed.
    """
    nrows, ncols = puzzle.shape
    i, j = slot
    
    if not is_in_grid(puzzle, i, j):
        # Slot is outside puzzle - try to roll
        if i < 0:
            if not np.all(puzzle[-1] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, 1, 0)
            i += 1
        elif i >= nrows:
            if not np.all(puzzle[0] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, -1, 0)
            i -= 1
        elif j < 0:
            if not np.all(puzzle[:, -1] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, 1, 1)
            j += 1
        elif j >= ncols:
            if not np.all(puzzle[:, 0] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, -1, 1)
            j -= 1
        else:
            raise ValueError('invalid slot')

    # Update
    unallocated_parts = unallocated_parts.copy()
    unallocated_parts.remove(part)
    puzzle[i][j] = part
    return puzzle, unallocated_parts


def place_remaining_parts(puzzle: np.ndarray, compatibility_matrix: np.ndarray, 
                         best_neighbours: np.ndarray, 
                         unallocated_parts: set) -> Tuple[np.ndarray, set]:
    """Place one remaining part."""
    candidate_slots = find_candidate_slots(puzzle)

    while True:
        matches = [(slot, part) for slot in candidate_slots for part in unallocated_parts 
                  if does_part_fit_in_slot(puzzle, best_neighbours, slot, part)]
        
        if len(matches) == 1:
            slot, part = matches.pop()
        else:
            average_compatibilities = [
                (average_compatibility_with_slot(puzzle, compatibility_matrix, slot, part), (slot, part))
                for slot in candidate_slots for part in unallocated_parts
            ]
            best = max(average_compatibilities, key=lambda x: x[0])
            slot, part = best[1]

        try:
            puzzle, unallocated_parts = try_assign(puzzle, slot, part, unallocated_parts)
            return puzzle, unallocated_parts
        except SlotAssignError:
            candidate_slots.remove(slot)
            if not candidate_slots:
                raise ValueError('no more slots')


def placer(solution: np.ndarray, compatibility_matrix: np.ndarray, 
          best_neighbours: np.ndarray, unallocated_parts: set) -> np.ndarray:
    """Place all remaining parts."""
    while unallocated_parts:
        solution, unallocated_parts = place_remaining_parts(
            solution, compatibility_matrix, best_neighbours, unallocated_parts
        )
    return solution


def calculate_best_buddies_metric(puzzle: np.ndarray, best_neighbours: np.ndarray) -> float:
    """
    Calculate best buddies metric (ratio of edges with best buddies).
    """
    nrows, ncols = puzzle.shape
    num_edges = (nrows - 1) * ncols + (ncols - 1) * nrows
    num_best_buddies = 0

    # Left/right edges
    for i in range(nrows):
        for j in range(ncols - 1):
            if best_buddies(best_neighbours, RIGHT, puzzle[i][j], puzzle[i][j + 1]):
                num_best_buddies += 1
    
    # Up/down edges
    for i in range(nrows - 1):
        for j in range(ncols):
            if best_buddies(best_neighbours, DOWN, puzzle[i][j], puzzle[i + 1][j]):
                num_best_buddies += 1

    return num_best_buddies / num_edges


def is_part_in_segment(puzzle: np.ndarray, best_neighbours: np.ndarray, 
                       segments: np.ndarray, segment: int, i: int, j: int) -> bool:
    """Check if a piece should be in a segment (all segment neighbors are best buddies)."""
    for relation in range(4):
        x, y = related_coords(relation, i, j)
        if is_in_grid(segments, x, y) and segments[x][y] == segment:
            if not best_buddies(best_neighbours, relation, puzzle[i][j], puzzle[x][y]):
                return False
    return True


def segment(puzzle: np.ndarray, best_neighbours: np.ndarray) -> np.ndarray:
    """
    Segment the puzzle into connected components based on best buddy relationships.
    """
    nrows, ncols = puzzle.shape
    segments = np.zeros((nrows, ncols), dtype=int)
    segment_counter = 1

    while True:
        unassigned_coords = np.argwhere(segments == 0)
        if unassigned_coords.size == 0:
            break

        # Start new segment from random unassigned piece
        stack = [unassigned_coords[np.random.choice(range(len(unassigned_coords)))]]
        while stack:
            i, j = stack.pop()
            segments[i][j] = segment_counter
            for x, y in adjacent(i, j):
                if is_in_grid(segments, x, y):
                    if segments[x][y] == 0 and is_part_in_segment(
                        puzzle, best_neighbours, segments, segment_counter, x, y
                    ):
                        stack.append((x, y))
        segment_counter += 1
    
    return segments


def largest_segment_index(segments: np.ndarray) -> int:
    """Find the index of the largest segment."""
    segment_indices, counts = np.unique(segments, return_counts=True)
    return segment_indices[np.argmax(counts)]


def mask_largest_segment(puzzle: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Keep only the largest segment, mask out others."""
    return np.where(segments == largest_segment_index(segments), puzzle, -1)


def center_occupied_in_grid(puzzle: np.ndarray) -> Tuple[np.ndarray, set]:
    """Center the occupied region in the grid."""
    nrows, ncols = puzzle.shape
    rows, cols = np.nonzero(puzzle + 1)
    delta_y = nrows // 2 - (max(rows) + min(rows)) // 2 - 1
    delta_x = ncols // 2 - (max(cols) + min(cols)) // 2 - 1
    puzzle = np.roll(puzzle, delta_y, 0)
    puzzle = np.roll(puzzle, delta_x, 1)
    allocated_parts = set(puzzle[puzzle != -1])
    unallocated_parts = set(range(nrows * ncols)) - allocated_parts
    return puzzle, unallocated_parts


def solve_puzzle(squares: List[np.ndarray], nrows: int, ncols: int, seed: int = None) -> Tuple[np.ndarray, float, int]:
    """
    Solve a jigsaw puzzle using the greedy algorithm.
    
    Args:
        squares: List of puzzle pieces in LAB color space, each (H, W, C)
        nrows: Number of rows in puzzle
        ncols: Number of columns in puzzle
        seed: Starting piece index (if None, will be auto-selected)
    
    Returns:
        solution: 2D array of piece indices
        best_score: Best buddies metric score
        iterations: Number of iterations
    """
    # Build compatibility matrix
    compatibility_matrix = build_compatibility_matrix(squares)
    best_neighbours = calculate_best_neighbours(compatibility_matrix)
    
    # Initialize solution
    if seed is None:
        seed = find_best_estimated_seed(best_neighbours)
    
    solution = np.full((nrows, ncols), -1)
    unallocated_parts = set(range(nrows * ncols))
    solution[nrows // 2][ncols // 2] = seed
    unallocated_parts.remove(seed)

    best_score = -1
    best_solution = None
    iterations = 0
    
    while True:
        # Place remaining pieces
        solution = placer(solution, compatibility_matrix, best_neighbours, unallocated_parts.copy())
        score = calculate_best_buddies_metric(solution, best_neighbours)
        
        if score <= best_score:
            break

        iterations += 1
        best_score = score
        best_solution = solution.copy()

        # Segment and keep largest segment
        segments = segment(solution, best_neighbours)
        masked = mask_largest_segment(solution, segments)

        # Re-center
        solution, unallocated_parts = center_occupied_in_grid(masked)

    return best_solution, best_score, iterations
