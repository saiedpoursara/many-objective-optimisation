"""
Simulation framework for interactive many-objective optimisation
using multiple reference points.

Test problem: DTLZ4
Author: Sarah Saeidpour
 """

import math
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Problem configurations
# Uncomment the block corresponding to the active number of objectives.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3 objectives
# ---------------------------------------------------------------------------
# num_objectives = 3              # Number of objectives
# num_xm_variables = 13           # Number of variables in DTLZ4 X_M
# num_variables = num_objectives + num_xm_variables - 1
#                                 # Number of decision variables
# num_extreme_solutions = 3       # Number of extreme Pareto-optimal solutions
#                                 # used to form the new pseudo payoff table

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0)
# dm_goal = (0.05, 0.10, 0.12)


# ---------------------------------------------------------------------------
# 5 objectives & 50 decision variables
# ---------------------------------------------------------------------------
# num_objectives = 5              # Number of objectives
# num_xm_variables = 46           # Number of variables in DTLZ4 X_M
# num_variables = num_objectives + num_xm_variables - 1
#                                 # Number of decision variables
# num_extreme_solutions = 5       # Number of extreme Pareto-optimal solutions
#                                 # used to form the new pseudo payoff table

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0)
# dm_goal = (0.05, 0.10, 0.12, 0.15, 0.12)


# ---------------------------------------------------------------------------
# 7 objectives
# ---------------------------------------------------------------------------
# num_objectives = 7              # Number of objectives
# num_xm_variables = 8            # Number of variables in DTLZ4 X_M
# num_variables = num_objectives + num_xm_variables - 1
#                                 # Number of decision variables
# num_extreme_solutions = 7       # Number of extreme Pareto-optimal solutions
#                                 # used to form the new pseudo payoff table

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5)
# dm_goal = (0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05)


# ---------------------------------------------------------------------------
# 9 objectives
# ---------------------------------------------------------------------------
# num_objectives = 9              # Number of objectives
# num_xm_variables = 10           # Number of variables in DTLZ4 X_M
# num_variables = num_objectives + num_xm_variables - 1
#                                 # Number of decision variables
# num_extreme_solutions = 9       # Number of extreme Pareto-optimal solutions
#                                 # used to form the new pseudo payoff table

# Simulated DM values for test purposes
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 8.0, 4.0)
# dm_goal = (0.05, 0.10, 0.12, 0.20, 0.01, 0.25, 0.06, 0.03, 0.18)


# ---------------------------------------------------------------------------
# 15 objectives
# ---------------------------------------------------------------------------
# num_objectives = 15             # Number of objectives
# num_xm_variables = 16           # Number of variables in DTLZ4 X_M
# num_variables = num_objectives + num_xm_variables - 1
#                                 # Number of decision variables
# num_extreme_solutions = 15      # Number of extreme Pareto-optimal solutions
#                                 # used to form the new pseudo payoff table

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
num_objectives = 20               # Number of objectives
num_xm_variables = 21             # Number of variables in DTLZ4 X_M
num_variables = num_objectives + num_xm_variables - 1
                                   # Number of decision variables
num_extreme_solutions = 20         # Number of extreme Pareto-optimal solutions
                                   # used to form the new pseudo payoff table

# Simulated DM values for test purposes
dm_goal_weights = (
    5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0, 6.0, 7.0,
    8.0, 9.0, 7.5, 6.5, 5.0, 6.0, 7.0, 8.0, 9.0, 7.5
)
dm_goal = (
    0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05, 0.12, 0.05, 0.10,
    0.12, 0.15, 0.12, 0.10, 0.05, 0.10, 0.12, 0.15, 0.12, 0.10
)


# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------
offset_distance = 1               # Offset distance
num_simulations = 5               # Number of simulations for each run case
num_rounds = 5                    # Number of rounds, including the initial round

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = (
    f"Method1Sims_DTLZ4_{num_objectives}Obj_{timestamp}.txt"
)

ideal = [-1 for _ in range(num_objectives)]
nadir = [-1 for _ in range(num_objectives)]


# ---------------------------------------------------------------------------
# Payoff table
# ---------------------------------------------------------------------------

# Pseudo payoff table: Instance 1
# Rotating pattern between rows (Stewart)
payoff_table = np.arange(num_objectives)
payoff_table = np.repeat(payoff_table, num_objectives)
payoff_table = np.reshape(
    payoff_table, (num_objectives, num_objectives)
)

for i in range(num_objectives):
    payoff_table[:, i] = np.roll(payoff_table[:, i], i)

payoff_table = payoff_table / (num_objectives - 1)


# Payoff table for testing the 3-objective case with known extreme points,
# i.e. a known conventional payoff table.
# payoff_table = np.zeros((3, 3))
# payoff_table[0, :] = [0, 0, 1]
# payoff_table[1, :] = [1, 0, 0]
# payoff_table[2, :] = [0, 1, 0]


# Ideal and nadir objective values derived from the payoff table
ideal = np.min(payoff_table, axis=0)
nadir = np.max(payoff_table, axis=0)


# ---------------------------------------------------------------------------
# GA Settings
# ---------------------------------------------------------------------------
population_size = 200              # GA population size
num_children = 200                 # Number of children
num_generations = 250              # Number of generations
binary_string_length = 10          # Length of binary representation
crossover_probability = 1.0        # Crossover probability
mutation_probability = 0.025       # Mutation probability
augmentation_parameter = 0.01      # Augmentation term parameter
num_reference_points = 7           # Number of reference points

num_good = num_reference_points // 3       # Number of Good solutions
num_moderate = num_reference_points // 2   # Number of Moderate solutions
num_poor = num_reference_points // 3       # Number of Poor solutions

# Conversion factors for mapping binary strings to real values
binary_to_real = [
    (2 ** (i - 1)) / (2 ** binary_string_length - 1)
    for i in range(binary_string_length, 0, -1)
]


def generate_dirichlet_weight_vectors(
    num_reference_points, num_extreme_solutions
):
    """Generate well-spread weight vectors using a Dirichlet distribution."""

    alpha = np.ones(num_extreme_solutions)
    weight_samples = np.random.dirichlet(
        alpha, size=num_reference_points * 20
    )
    labels = [f"s{i + 1}" for i in range(num_reference_points * 20)]
    weight_samples_df = pd.DataFrame(weight_samples, index=labels)

    distances = np.full(
        (num_reference_points * 20, num_reference_points * 20),
        2,
        dtype=float,
    )
    distances_df = pd.DataFrame(
        distances, index=labels, columns=labels
    )

    for i in range(num_reference_points * 20):
        for j in range(i, num_reference_points * 20):
            distance = 0
            for k in range(num_extreme_solutions):
                distance += (
                    weight_samples[i, k] - weight_samples[j, k]
                ) ** 2
            distances_df.iloc[i, j] = math.sqrt(distance)

    for i in range(num_reference_points * 20):
        for j in range(num_reference_points * 20):
            if i < j:
                distances_df.iloc[j, i] = distances_df.iloc[i, j]

    for i in range(num_reference_points * 20):
        distances_df.iloc[i, i] = 2

    remaining_labels = labels.copy()

    for _ in range(num_reference_points * 19):
        closest_pair = np.unravel_index(
            np.nanargmin(distances_df.values), distances_df.shape
        )

        min_distance_2 = distances_df[
            remaining_labels[closest_pair[1]]
        ].min()
        min_distance_1 = distances_df.loc[
            remaining_labels[closest_pair[0]], :
        ].min(axis=0)

        if min_distance_2 >= min_distance_1:
            distances_df.drop(
                labels=remaining_labels[closest_pair[0]],
                axis=0,
                inplace=True,
            )
            distances_df.drop(
                labels=remaining_labels[closest_pair[0]],
                axis=1,
                inplace=True,
            )
        else:
            distances_df.drop(
                labels=remaining_labels[closest_pair[1]],
                axis=0,
                inplace=True,
            )
            distances_df.drop(
                labels=remaining_labels[closest_pair[1]],
                axis=1,
                inplace=True,
            )

        remaining_labels.remove(remaining_labels[closest_pair[0]])

    final_weight_sample = pd.DataFrame(
        weight_samples_df, index=remaining_labels
    )

    return pd.DataFrame(final_weight_sample).to_numpy()


def evaluate_dtlz4(x):
    """Evaluate the DTLZ4 objective functions."""

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

    return objective_values
 
    
def wierzbicki_asf(objective_values, reference_points, num_reference_points):
    """Calculate scalarising function values for all reference points."""

    scalarising_function_values = []

    for r in range(num_reference_points):
        reference_point = reference_points[r]
        deviations = objective_values - reference_point
        scalarising_function_value = (
            max(deviations) + 0.01 * sum(deviations)
        )
        scalarising_function_values.append(scalarising_function_value)

    return scalarising_function_values


def run_ga(reference_points):
    """Run the GA using multiple reference points."""

    # Functions for ranking the population
    def rank_inner_2(j, c, population_size):
        for k in range(c, population_size - 1):
            if sorted_indices[k + 1, j] != -1:
                append_to_rank(sorted_indices[k + 1, j])
                sorted_indices[
                    sorted_indices == sorted_indices[k + 1, j]
                ] = -1
                c = k + 1
                break

    def rank_inner_1(c, i, num_reference_points, population_size):
        for j in range(num_reference_points):
            if len(population_rank) == population_size:
                break

            if sorted_indices[c, j] != -1:
                append_to_rank(sorted_indices[c, j])
            else:
                rank_inner_2(j, c, population_size)

            sorted_indices[sorted_indices == sorted_indices[c, j]] = -1
            c = i

    def rank_population(population_size):
        for i in range(population_size):
            if len(population_rank) < population_size:
                c = i
                rank_inner_1(
                    c, i, num_reference_points, population_size
                )
            else:
                break
            
    # Generate the initial population randomly as binary strings.
    population = np.random.randint(
        2,
        size=(
            population_size,
            num_variables,
            binary_string_length,
        ),
    )

    # Decode the binary strings into real-valued decision variables.
    population_decision_variables = np.dot(
        population, binary_to_real
    )

    # Evaluate the objective functions for the population.
    population_objective_values = np.apply_along_axis(
        evaluate_dtlz4, 1, population_decision_variables
    )

    # Calculate ASF values for each population member and RP.
    population_asf = np.apply_along_axis(
        wierzbicki_asf,
        1,
        population_objective_values,
        reference_points,
        num_reference_points,
    )

    # Rank the population based on the ASF values of multiple RPs
    # simultaneously.
    population_rank = []
    append_to_rank = population_rank.append
    sorted_indices = np.argsort(population_asf, axis=0)

    rank_population(population_size)

    rank = np.array(population_rank)
    population = population[rank]
    population_objective_values = population_objective_values[rank]
    population_asf = population_asf[rank]

    # Store the best ASF value for each RP in each generation.
    asf_trace = np.empty(
        [num_generations + 1, num_reference_points]
    )
    asf_trace[0, :] = population_asf[0]
    
    # Run the generations.
    for generation in range(num_generations):
        parent_1_sample = np.random.choice(
            population_size, (4, num_children)
        )
        parent_1 = np.amin(parent_1_sample, axis=0)

        parent_2_sample = np.random.choice(
            population_size, (4, num_children)
        )
        parent_2 = np.amin(parent_2_sample, axis=0)

        # Select a parent independently for each gene.
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

        # Apply mutation.
        mutation = np.random.binomial(
            1,
            mutation_probability,
            (num_children, num_variables, binary_string_length),
        )
        children = (
            mutation
            + children
            - 2 * mutation * children
        )
    
                # Decode the children into real-valued decision variables.
        children_decision_variables = np.dot(
            children, binary_to_real
        )

        # Evaluate the objective functions for the children.
        children_objective_values = np.apply_along_axis(
            evaluate_dtlz4, 1, children_decision_variables
        )

        # Calculate ASF values for each child and RP.
        children_asf = np.apply_along_axis(
            wierzbicki_asf,
            1,
            children_objective_values,
            reference_points,
            num_reference_points,
        )

        # Combine the parent and child populations.
        population = np.concatenate((population, children))
        population_objective_values = np.concatenate(
            (population_objective_values, children_objective_values)
        )
        population_asf = np.concatenate(
            (population_asf, children_asf)
        )
    
                # Rank the combined parent and child populations based on the
        # ASF values of multiple RPs simultaneously.
        sorted_indices = np.argsort(population_asf, axis=0)
        population_rank = []
        append_to_rank = population_rank.append

        rank_population(population_size + num_children)

        rank = np.array(population_rank)
        population = population[rank]
        population_asf = population_asf[rank]
        population_objective_values = population_objective_values[rank]

        # Prune back to the original population size.
        population = population[:population_size]
        population_asf = population_asf[:population_size]
        population_objective_values = population_objective_values[
            :population_size
        ]

        asf_trace[generation + 1, :] = population_asf[0]
        
    plt.plot(asf_trace[:250, :])
    plt.legend(
        [f"RP{i + 1}" for i in range(num_reference_points)]
    )
    plt.xlabel("Generations")
    plt.ylabel("Minimum ASF values")
    plt.show()

    return population_objective_values[:num_reference_points]    

#---------------------------------MAIN CODE---------------------------------------------------
start=datetime.now() 
# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------
log_file = open(log_filename, "w")

log_file.write(
    "A GA for Multiple Reference Points Optimisation on DTLZ4.\n"
)
log_file.write("Method 1: A Geometric Approach\n")
log_file.write(
    datetime.now().strftime("%Y-%m-%d %H:%M") + "\n\n"
)

log_file.write(
    "No. of objectives          : {:2d}\n".format(num_objectives)
)
log_file.write(
    "No. of vars in DTLZ4 X_M   : {:2d}\n".format(num_xm_variables)
)
log_file.write(
    "GA pop size                : {:4d}\n".format(population_size)
)
log_file.write(
    "GA no. of children         : {:4d}\n".format(num_children)
)
log_file.write(
    "GA no. of generations      : {:4d}\n".format(num_generations)
)
log_file.write(
    "GA mutation probability    : {:6.3f}\n".format(
        mutation_probability
    )
)
log_file.write(
    "Length of binary precision : {:3d}\n".format(
        binary_string_length
    )
)
log_file.write(
    "Offset distance            : {:5.2f}\n\n\n".format(
        offset_distance
    )
)

# ---------------------------------------------------------------------------
# Minimum weighted distance from the DM's goal to the true Pareto frontier.
# Uncomment the line corresponding to the active problem configuration.
# ---------------------------------------------------------------------------

# log_file.write(
#     "*** The minimum weighted distance from the defined DM's goal, "
#     "achievable on the true Pareto frontier for this problem is 4.29 "
#     "(and 4.08 by differential_evolution package, and also 4.08 by our GA).\n\n\n"
# )  # 3 objectives, 15 variables

# log_file.write(
#     "*** The minimum weighted distance from the defined DM's goal, "
#     "achievable on the true Pareto frontier for this problem is 3.66 "
#     "(and 3.6 by differential_evolution package, and 3.55 by our GA).\n\n\n"
# )  # 7 objectives

# log_file.write(
#     "*** The minimum weighted distance from the defined DM's goal, "
#     "achievable on the true Pareto frontier for this problem is 3.69 "
#     "(and 3.69 by differential_evolution package, and 3.99 by our GA).\n\n\n"
# )  # 5 objectives, 50 variables

# log_file.write(
#     "*** The minimum weighted distance from the defined DM's goal, "
#     "achievable on the true Pareto frontier for this problem is 1.93 "
#     "(and 1.95 by differential_evolution package, and 1.94 by our GA).\n\n\n"
# )  # 9 objectives

# log_file.write(
#     "*** The minimum weighted distance from the defined DM's goal, "
#     "achievable on the true Pareto frontier for this problem is 2.35 "
#     "(and 2.41 by differential_evolution package, and 2.45 by our GA).\n\n\n"
# )  # 15 objectives

log_file.write(
    "*** The minimum weighted distance from the defined DM's goal, "
    "achievable on the true Pareto frontier for this problem is 1.83 "
    "(and 1.89 by differential_evolution package, and 2.13 by our GA).\n\n\n"
)  # 20 objectives


# ---------------------------------------------------------------------------
# Interaction rounds
# ---------------------------------------------------------------------------  
weight_vectors = generate_dirichlet_weight_vectors(
    num_reference_points, num_extreme_solutions
)
extreme_solutions = payoff_table
log_file.write("Payoff Table of the problem:\n")

for f in range(num_objectives):
    log_file.write("f{}(x).".format(f + 1))
    for j in range(num_objectives):
        log_file.write(
            "{:10.3f}".format(extreme_solutions[f][j])
        )
    log_file.write("\n")

log_file.write("\n")
log_file.write(
    "The Ideal and Nadir points from the Payoff Table:\n"
)

log_file.write("Ideal point: ")
for i in range(num_objectives):
    log_file.write("{:10.3f}".format(ideal[i]))
log_file.write("\n")

log_file.write("Nadir point:")
for i in range(num_objectives):
    log_file.write("{:10.3f}".format(nadir[i]))
log_file.write("\n\n")

log_file.write(
    "The Euclidean distance between the Ideal and the Nadir points "
    "(DoSS)             : {:5.3f}\n".format(
        math.dist(ideal, nadir)
    )
)

log_file.write(
    "________________________________________________________________"
    "________________________________________________________\n\n"
)    

# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------
simulation_average_best_values = []
simulation_low = []
simulation_high = []
simulation_psh = []
simulation_psh_rp = []
simulation_final_round_range_low = []
simulation_final_round_range_high = []

doss = math.dist(ideal, nadir)

for simulation in range(num_simulations):
    log_file.write(
        "\nSimulation{}:\n".format(simulation + 1)
    )
    log_file.write(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )
    log_file.write(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )

    # Largest Euclidean distance between PO solutions of the
    # previous iteration.
    largest_distance_pi = doss

    best_of_round = []
    worst_of_round = []
    median_of_round = []

    num_extreme_solutions = num_objectives
    weight_vectors = generate_dirichlet_weight_vectors(
        num_reference_points, num_extreme_solutions
    )
    extreme_solutions = payoff_table
    
    for round_index in range(num_rounds):

        # Generate RPs as weighted combinations of the extreme solutions.
        reference_points = np.zeros(
            [num_reference_points, num_objectives]
        )

        for i in range(num_reference_points):
            reference_points[i, :] = sum(
                weight_vectors[i, j] * extreme_solutions[j, :]
                for j in range(num_extreme_solutions)
            )

        log_file.write(
            "\nRound{}:\n".format(round_index + 1)
        )
        log_file.write("Reference points as row vectors:\n")

        for r in range(num_reference_points):
            log_file.write("RP.{}     ".format(r + 1))
            for j in range(num_objectives):
                log_file.write(
                    "{:10.3f}".format(reference_points[r][j])
                )
            log_file.write("\n")
        
        # Smallest and largest Euclidean distances between RPs.
        distances = pdist(reference_points)
        sorted_distance_indices = np.argsort(distances)
        reference_point_pairs = list(
            combinations(range(num_reference_points), 2)
        )

        # Two farthest RPs.
        farthest_pair_index = sorted_distance_indices[-1]
        two_farthest = reference_point_pairs[farthest_pair_index]
        largest_distance = distances[farthest_pair_index]

        # Two closest RPs.
        closest_pair_index = sorted_distance_indices[0]
        two_closest = reference_point_pairs[closest_pair_index]
        smallest_distance = distances[closest_pair_index]

        log_file.write(
            "The Euclidean distance between two farthest Reference "
            "Points is    : {:5.3f}\n".format(largest_distance)
        )
        log_file.write(
            "The Euclidean distance between two closest Reference "
            "Points is     : {:5.3f}\n".format(smallest_distance)
        )
        
        po_solutions = run_ga(reference_points)

        # Smallest and largest Euclidean distances between PO solutions.
        po_distances = pdist(po_solutions)
        sorted_po_distance_indices = np.argsort(po_distances)
        po_solution_pairs = list(
            combinations(range(num_reference_points), 2)
        )

        # Two farthest PO solutions.
        farthest_po_pair_index = sorted_po_distance_indices[-1]
        two_farthest_po = po_solution_pairs[farthest_po_pair_index]
        largest_distance_po = po_distances[farthest_po_pair_index]

        # Two closest PO solutions.
        closest_po_pair_index = sorted_po_distance_indices[0]
        two_closest_po = po_solution_pairs[closest_po_pair_index]
        smallest_distance_po = po_distances[closest_po_pair_index]

        # Percentage Change in Search Space in the Current Iteration
        # (RSSCI).
        reduction_search_space_current_iteration = (
            largest_distance_po - largest_distance_pi
        )
        percentage_reduction_search_space_current_iteration = (
            reduction_search_space_current_iteration
            / largest_distance_pi
        ) * 100

        largest_distance_pi = largest_distance_po
        
        # Percentage Change in RP Space in the Current Iteration
        # (RRPSCI).
        if round_index == 0:
            reduction_rp_space_current_iteration = 0
            percentage_reduction_rp_space_current_iteration = 0
            largest_distance_rp_pi = largest_distance
            largest_distance_rp_first_iteration = largest_distance
        else:
            reduction_rp_space_current_iteration = (
                largest_distance - largest_distance_rp_pi
            )
            percentage_reduction_rp_space_current_iteration = (
                reduction_rp_space_current_iteration
                / largest_distance_rp_pi
            ) * 100

        # Largest Euclidean distance between RPs of the previous iteration.
        largest_distance_rp_pi = largest_distance
        
        log_file.write("\n")
        log_file.write(
            "The Euclidean distance between two farthest Pareto "
            "Optimal solutions is    : {:5.3f}\n".format(
                largest_distance_po
            )
        )
        log_file.write(
            "The Euclidean distance between two closest Pareto "
            "Optimal solutions is     : {:5.3f}\n".format(
                smallest_distance_po
            )
        )

        # Weighted sum of squares of deviations above the DM's goal.
        distance_from_goal = np.zeros(num_reference_points)

        for r in range(num_reference_points):
            distance_from_goal[r] = sum(
                dm_goal_weights[j]
                * max(
                    0,
                    po_solutions[r, j] - dm_goal[j],
                ) ** 2
                for j in range(num_objectives)
            )
            
        # Euclidean distance between RPs and their corresponding PO solutions.
        distance_rp_to_po_solution = np.zeros(num_reference_points)

        for i in range(num_reference_points):
            distance_rp_to_po_solution[i] = math.dist(
                reference_points[i, :],
                po_solutions[i, :],
            )

        log_file.write(
            "\nWeighted distances of PO solutions corresponding to each RP, "
            "from the DM's Goal, and their Euclidean distances from their "
            "corresponding RP:\n\n"
        )
        log_file.write(
            "                                  {:^25}{:^25}\n".format(
                "W_Dis from Goal", "E_Dis from RP"
            )
        )

        for r in range(num_reference_points):
            log_file.write(
                "PO Solution of RP.{}     ".format(r + 1)
            )
            log_file.write(
                "{:^30.3f}{:^30.3f}".format(
                    distance_from_goal[r],
                    distance_rp_to_po_solution[r],
                )
            )
            log_file.write("\n")     
            
        best_of_round.append(distance_from_goal.min())
        worst_of_round.append(distance_from_goal.max())
        median_of_round.append(np.median(distance_from_goal))

        log_file.write(
            "\nBest/worst/median weighted distance values of round{}:   "
            "B={:<8.3f}  W={:<8.3f}  M={:<8.3f}\n".format(
                round_index + 1,
                distance_from_goal.min(),
                distance_from_goal.max(),
                np.median(distance_from_goal),
            )
        )

        log_file.write(
            "\nPercentage Change in the search space in round{}:   "
            "%{:<8.3f}\n".format(
                round_index + 1,
                percentage_reduction_search_space_current_iteration,
            )
        )

        log_file.write(
            "\nPercentage Change in the RP space in round{}:   "
            "%{:<8.3f}\n".format(
                round_index + 1,
                percentage_reduction_rp_space_current_iteration,
            )
        )

        log_file.write(
            "................................................................"
            "...................................................\n"
        )

        # Reference point design
        distance_from_goal_indices = np.argsort(distance_from_goal)

        centre_point = [
            sum(reference_points) / num_reference_points
        ]

        good_reference_points = np.array(
            [
                reference_points[i]
                for i in distance_from_goal_indices[:num_good]
            ]
        )
        moderate_reference_points = np.array(
            [
                reference_points[i]
                for i in distance_from_goal_indices[
                    num_good:num_good + num_moderate
                ]
            ]
        )
        poor_reference_points = np.array(
            [
                reference_points[i]
                for i in distance_from_goal_indices[-num_poor:]
            ]
        )

        good_reference_points = (
            centre_point
            + (1 + offset_distance)
            * (good_reference_points - centre_point)
        )
        poor_reference_points = (
            centre_point
            + (1 - offset_distance)
            * (poor_reference_points - centre_point)
        )

        # Form the new pseudo payoff table.
        extreme_solutions = np.concatenate(
            (
                good_reference_points,
                moderate_reference_points,
                poor_reference_points,
            )
        )

        num_extreme_solutions = extreme_solutions.shape[0]

        weight_vectors = generate_dirichlet_weight_vectors(
            num_reference_points, num_extreme_solutions
        )
        
    # Summary results for each simulation    
    average_best_value = sum(best_of_round) / num_rounds
    low = min(best_of_round)
    high = max(best_of_round)

    percentage_search_space_reduction = (
        (largest_distance_po - math.dist(ideal, nadir))
        / math.dist(ideal, nadir)
    ) * 100

    percentage_rp_space_reduction = (
        (largest_distance - largest_distance_rp_first_iteration)
        / largest_distance_rp_first_iteration
    ) * 100

    log_file.write(
        "\n____________________________________________________________"
        "__________________________________________________________\n"
    )
    log_file.write(
        "\n____________________________________________________________"
        "__________________________________________________________\n\n"
    )

    log_file.write(
        "\nResult summary of simulation {}:\n".format(
            simulation + 1
        )
    )
    log_file.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    log_file.write(
        "\nMean best weighted distance values: {:8.3f}\n".format(
            average_best_value
        )
    )
    log_file.write(
        "\nRange of weighted distance values in the final round: "
        "{:8.3f}   -   {:8.3f}\n".format(
            distance_from_goal.min(),
            distance_from_goal.max(),
        )
    )
    log_file.write(
        "\nRange of best weighted distance values: "
        "{:8.3f}  -{:8.3f}\n".format(low, high)
    )
    log_file.write(
        "\nPercentage Change in the search space in the whole "
        "process:   %{:<8.3f}\n".format(
            percentage_search_space_reduction
        )
    )
    log_file.write(
        "\nPercentage Change in the RP space in the whole "
        "process:   %{:<8.3f}\n".format(
            percentage_rp_space_reduction
        )
    )
    #-------------------------------------------------------------------------
    end = datetime.now()
    duration = end - start
    milliseconds = (
        duration.seconds * 1000
        + duration.microseconds / 1000
    )

    log_file.write(
        "\n\nDone in {} minutes, {} seconds, {} milliseconds.\n".format(
            int(duration.seconds / 60),
            duration.seconds % 60,
            milliseconds % 1000,
        )
    )
    log_file.write(
        "\n____________________________________________________________"
        "__________________________________________________________\n\n"
    )

    print()
    print(
        "Simulation {} completed. Cumulative runtime: "
        "{} minutes, {} seconds, {:.0f} milliseconds.".format(
            simulation + 1,
            int(duration.seconds / 60),
            duration.seconds % 60,
            milliseconds % 1000,
        )
    )
    
    log_file.write(
        "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )
    log_file.write(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )

    simulation_average_best_values.append(average_best_value)
    simulation_low.append(low)
    simulation_high.append(high)
    simulation_psh.append(percentage_search_space_reduction)
    simulation_psh_rp.append(percentage_rp_space_reduction)
    simulation_final_round_range_low.append(
        distance_from_goal.min()
    )
    simulation_final_round_range_high.append(
        distance_from_goal.max()
    )

log_file.write(
    "\n************************************************************"
    "***************************************************\n"
)
log_file.write(
    "************************************************************"
    "*****************************************************\n\n"
)

log_file.write(
    "Result summary of {} simulations:\n\n".format(
        num_simulations
    )
)

log_file.write(
    "\nMean best weighted distance values of {} simulations: "
    "{:8.3f}\n".format(
        num_simulations,
        sum(simulation_average_best_values) / num_simulations,
    )
)

log_file.write(
    "\nAverage range of best weighted distance values in {} "
    "simulations: {:8.3f}  -{:8.3f}\n".format(
        num_simulations,
        sum(simulation_low) / num_simulations,
        sum(simulation_high) / num_simulations,
    )
)

log_file.write(
    "\nAverage range of best weighted distance values in the final "
    "round in {} simulations: {:8.3f}  -{:8.3f}\n".format(
        num_simulations,
        sum(simulation_final_round_range_low) / num_simulations,
        sum(simulation_final_round_range_high) / num_simulations,
    )
)

log_file.write(
    "\nAverage percentage Change in the search space in the whole "
    "process, in {} simulations:   %{:<8.3f}\n".format(
        num_simulations,
        sum(simulation_psh) / num_simulations,
    )
)

log_file.write(
    "\nAverage percentage Change in the RP space in the whole "
    "process, in {} simulations:   %{:<8.3f}\n".format(
        num_simulations,
        sum(simulation_psh_rp) / num_simulations,
    )
)

log_file.close()
