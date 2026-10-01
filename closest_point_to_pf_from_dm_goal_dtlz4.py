"""
Calculate the closest point on the DTLZ4 Pareto front to a simulated DM goal.

Author: Sarah Saeidpour
"""

import math

import numpy as np
from scipy.optimize import differential_evolution

  
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


# ---------------------------------------------------------------------------
# DTLZ4 evaluation and weighted distance from the DM goal
# ---------------------------------------------------------------------------
def evaluate_dtlz4(x):
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
# Optimisation
# ---------------------------------------------------------------------------
bounds = np.array([[0, 1] for _ in range(num_variables)])

result = differential_evolution(evaluate_dtlz4, bounds)

print(f"Minimum weighted distance: {result.fun}")
print(f"Decision vector: {result.x}")
