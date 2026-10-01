"""
Find the point on the DTLZ4 Pareto front closest to a simulated DM goal
using a binary-coded genetic algorithm.

Author: Sarah Saeid Pour
"""

import math

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Problem configurations
# Uncomment the block corresponding to the active number of objectives.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3 objectives
# ---------------------------------------------------------------------------
num_objectives = 3
num_xm_variables = 13
num_variables = num_objectives + num_xm_variables - 1

# Simulated DM values for test purposes
dm_goal_weights = (5.0, 6.0, 7.0)
dm_goal = (0.05, 0.10, 0.12)


# ---------------------------------------------------------------------------
# 5 objectives & 50 decision variables
# ---------------------------------------------------------------------------
# num_objectives = 5
# num_xm_variables = 46
# num_variables = num_objectives + num_xm_variables - 1

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0)
# dm_goal = (0.05, 0.10, 0.12, 0.15, 0.12)


# ---------------------------------------------------------------------------
# 7 objectives
# ---------------------------------------------------------------------------
# num_objectives = 7
# num_xm_variables = 8
# num_variables = num_objectives + num_xm_variables - 1

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5)
# dm_goal = (0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05)


# ---------------------------------------------------------------------------
# 9 objectives
# ---------------------------------------------------------------------------
# num_objectives = 9
# num_xm_variables = 10
# num_variables = num_objectives + num_xm_variables - 1

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 8.0, 4.0)
# dm_goal = (0.05, 0.10, 0.12, 0.20, 0.01, 0.25, 0.06, 0.03, 0.18)


# ---------------------------------------------------------------------------
# 15 objectives
# ---------------------------------------------------------------------------
# num_objectives = 15
# num_xm_variables = 16
# num_variables = num_objectives + num_xm_variables - 1

# Simulated DM values for test purposes
# dm_goal_weights = (
#     5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0,
#     6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0
# )
# dm_goal = (
#     0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05, 0.12,
#     0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05
# )


# ---------------------------------------------------------------------------
# 20 objectives
# ---------------------------------------------------------------------------
# num_objectives = 20
# num_xm_variables = 21
# num_variables = num_objectives + num_xm_variables - 1

# Simulated DM values for test purposes
# dm_goal_weights = (
#     5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0, 6.0, 7.0,
#     8.0, 9.0, 7.5, 6.5, 5.0, 6.0, 7.0, 8.0, 9.0, 7.5
# )
# dm_goal = (
#     0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05, 0.12, 0.05, 0.10,
#     0.12, 0.15, 0.12, 0.10, 0.05, 0.10, 0.12, 0.15, 0.12, 0.10
# )


bounds = np.array([[0, 1] for _ in range(num_variables)])


# ---------------------------------------------------------------------------
# GA settings
# ---------------------------------------------------------------------------
population_size = 200
num_children = 200
num_generations = 1000
binary_string_length = 10
crossover_probability = 1
mutation_probability = 0.025
augmentation_parameter = 0.01
num_reference_points = 1


# ---------------------------------------------------------------------------
# Binary decoding
# ---------------------------------------------------------------------------
# Weights used to convert each binary string to its corresponding real value
# in [0, 1], using the identity sum(2^(n - 1)) = 2^n - 1.
binary_to_real = [
    (2 ** (i - 1)) / (2 ** binary_string_length - 1)
    for i in range(binary_string_length, 0, -1)
]


# ---------------------------------------------------------------------------
# DTLZ4 evaluation and weighted distance from the DM goal
# ---------------------------------------------------------------------------
def evaluate_dtlz4(x):
    """Return the weighted distance of a DTLZ4 solution from the DM goal."""
    objective_values = []
    alpha = 100

    g_xm = 1
    for i in range(num_xm_variables):
        g_xm += (x[i + num_objectives - 1] - 0.5) ** 2

    objective_values.append(1)

    for i in range(1, num_objectives):
        objective_values.append(
            objective_values[i - 1]
            * math.cos((math.pi / 2) * (x[i - 1] ** alpha))
        )

    for i in range(num_objectives - 1):
        objective_values[i] *= math.sin(
            (math.pi / 2) * (x[i] ** alpha)
        )

    objective_values.reverse()
    objective_values = [
        g_xm * value for value in objective_values
    ]

    weighted_distance = 0
    for i in range(num_objectives):
        weighted_distance += (
            dm_goal_weights[i]
            * max(0, objective_values[i] - dm_goal[i]) ** 2
        )

    return weighted_distance


# ---------------------------------------------------------------------------
# GA ranking
# ---------------------------------------------------------------------------
def rankinn2(j, c, pop_size):
    for k in range(c, pop_size - 1):
        if sorted_indices[k + 1, j] != -1:
            append_rank(sorted_indices[k + 1, j])
            sorted_indices[
                sorted_indices == sorted_indices[k + 1, j]
            ] = -1
            break


def rankinn1(c, i, num_rps, pop_size):
    for j in range(num_rps):
        if len(population_rank) == pop_size:
            break

        if sorted_indices[c, j] != -1:
            append_rank(sorted_indices[c, j])
        else:
            rankinn2(j, c, pop_size)

        sorted_indices[
            sorted_indices == sorted_indices[c, j]
        ] = -1
        c = i


def rankout(pop_size):
    for i in range(pop_size):
        if len(population_rank) < pop_size:
            c = i
            rankinn1(c, i, num_reference_points, pop_size)
        else:
            break


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------
# Generate the first population randomly as binary strings.
population = np.random.randint(
    2,
    size=(
        population_size,
        num_variables,
        binary_string_length,
    ),
)

# Decode the binary strings to real-valued decision variables.
population_dv = np.dot(population, binary_to_real)

# Calculate the weighted distance of each population member from the DM goal.
population_fe = np.apply_along_axis(
    evaluate_dtlz4,
    1,
    population_dv,
)
population_fe = population_fe.reshape((population_size, 1))


# ---------------------------------------------------------------------------
# Initial population ranking
# ---------------------------------------------------------------------------
population_rank = []
append_rank = population_rank.append
sorted_indices = np.argsort(population_fe, axis=0)

rankout(population_size)

rank = np.array(population_rank)
population = population[rank]
population_fe = population_fe[rank]


# Store the best weighted distance in each generation.
weighted_distance_trace = np.empty(
    [num_generations + 1, num_reference_points]
)
weighted_distance_trace[0, :] = population_fe[0]
        
# ---------------------------------------------------------------------------
# Run generations
# ---------------------------------------------------------------------------
for generation in range(num_generations):
    # Select two parents using tournaments of size four. Since the population
    # is already ranked, the lowest sampled index represents the best entrant.
    parent_1_sample = np.random.choice(
        population_size, (4, num_children)
    )
    parent_1 = np.amin(parent_1_sample, axis=0)

    parent_2_sample = np.random.choice(
        population_size, (4, num_children)
    )
    parent_2 = np.amin(parent_2_sample, axis=0)

    # Select the parent of each gene independently to form the children.
    crossover = np.random.randint(
        2,
        size=(
            num_children,
            num_variables,
            binary_string_length,
        ),
    )
    children = (
        crossover * population[parent_1]
        + (1 - crossover) * population[parent_2]
    )

    # Apply bit-flip mutation.
    mutation = np.random.binomial(
        1,
        mutation_probability,
        (
            num_children,
            num_variables,
            binary_string_length,
        ),
    )
    children = (
        mutation
        + children
        - 2 * mutation * children
    )

    # Decode and evaluate the children.
    children_dv = np.dot(children, binary_to_real)
    children_fe = np.apply_along_axis(
        evaluate_dtlz4,
        1,
        children_dv,
    )
    children_fe = children_fe.reshape((num_children, 1))

    # Combine the parent and child populations.
    population = np.concatenate((population, children))
    population_fe = np.concatenate(
        (population_fe, children_fe)
    )

    # Rank the combined population.
    sorted_indices = np.argsort(population_fe, axis=0)
    population_rank = []
    append_rank = population_rank.append

    rankout(population_size + num_children)

    rank = np.array(population_rank)
    population = population[rank]
    population_fe = population_fe[rank]

    # Prune back to the original population size.
    population = population[:population_size]
    population_fe = population_fe[:population_size]

    weighted_distance_trace[generation + 1, :] = population_fe[0]
    
# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
plt.plot(weighted_distance_trace)
plt.xlabel("Generation")
plt.ylabel("Minimum weighted distance")
plt.show()

print(
    "\nMinimum weighted distance from the DM goal:",
    population_fe[0],
)

# Decode the best solution.
best_binary_solution = np.asarray(population[0, :, :])
best_solution = np.dot(best_binary_solution, binary_to_real)

print(
    "\nDecision vector for the closest point to the DM goal:",
    best_solution,
)   

 
# ---------------------------------------------------------------------------
# Objective-vector evaluation and Pareto-front check
# ---------------------------------------------------------------------------
def evaluate_solution(x):
    """Evaluate the objective vector used for the final PF check."""
    objective_values = []
    alpha = 100

    g_xm = 1
    for i in range(num_xm_variables):
        g_xm += (x[i + num_objectives - 1] - 0.5) ** 2

    objective_values.append(1)

    for i in range(1, num_objectives):
        objective_values.append(
            objective_values[i - 1]
            * math.cos((math.pi / 2) * (x[i - 1] ** alpha))
        )

    for i in range(num_objectives - 1):
        objective_values[i] *= math.sin(
            (math.pi / 2) * x[i]
        )

    objective_values.reverse()
    objective_values = [
        g_xm * value for value in objective_values
    ]

    return objective_values


objective_values = evaluate_solution(best_solution)

print("\nF(x) =", objective_values)

# Check the spherical PF condition.
pf_check = sum(
    value ** 2 for value in objective_values
)

print(
    "\nPF shape check (sum of squared objective values):",
    pf_check,
)