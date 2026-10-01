"""
Calculate the minimum weighted distance from a simulated DM goal
to hyperspherical and multiaxial ellipsoidal surfaces.

Test problems: DTLZ2, DTLZ4, WFG4 and WFG9
Author: Sarah Saeidpour
"""

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

# ---------------------------------------------------------------------------
# Problem configurations
# Uncomment the block corresponding to the active number of objectives
# and test problem.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3 objectives
# ---------------------------------------------------------------------------
# num_objectives = 3
# dm_goal_weights = (5.0, 6.0, 7.0)
# dm_goal = (0.05, 0.10, 0.12)  # DTLZ4
# dm_goal = (0.5, 0.3, 2.0)     # WFG4 and WFG9


# ---------------------------------------------------------------------------
# 5 objectives
# ---------------------------------------------------------------------------
# num_objectives = 5
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0)
# dm_goal = (0.05, 0.10, 0.12, 0.15, 0.12)  # DTLZ4
# dm_goal = (0.5, 0.3, 2.0, 2.4, 0.1)       # WFG4 and WFG9


# ---------------------------------------------------------------------------
# 7 objectives
# ---------------------------------------------------------------------------
# num_objectives = 7
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5)
# dm_goal = (0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05)  # DTLZ2 and DTLZ4
# dm_goal = (0.05, 0.2, 0.8, 0.42, 1.01, 2.06, 7.0)     # WFG4 and WFG9


# ---------------------------------------------------------------------------
# 9 objectives
# ---------------------------------------------------------------------------
# num_objectives = 9

# DTLZ4
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 8.0, 4.0)
# dm_goal = (0.05, 0.10, 0.12, 0.20, 0.01, 0.25, 0.06, 0.03, 0.18)

# WFG4 and WFG9
# dm_goal_weights = (5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0, 6.0)
# dm_goal = (0.05, 0.2, 0.8, 0.42, 0.01, 2.042, 3.01, 4.06, 5.0)


# ---------------------------------------------------------------------------
# 15 objectives
# ---------------------------------------------------------------------------
# num_objectives = 15
# dm_goal_weights = (
#     5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0,
#     6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0
# )
# dm_goal = (
#     0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05, 0.12,
#     0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05
# )  # DTLZ2 and DTLZ4
# dm_goal = (
#     0.5, 0.2, 0.8, 4.2, 1.01, 2.06, 7.0, 5.0,
#     3.0, 2.0, 2.042, 4.01, 6.7, 8.08, 10.05
# )  # WFG4
# dm_goal = (
#     0.5, 0.2, 0.8, 4.2, 1.01, 2.06, 7.0, 5.0,
#     3.0, 2.0, 2.042, 1.01, 1.7, 1.08, 1.05
# )  # WFG9


# ---------------------------------------------------------------------------
# 20 objectives
# ---------------------------------------------------------------------------
num_objectives = 20
dm_goal_weights = (
    5.0, 6.0, 7.0, 8.0, 9.0, 7.5, 6.5, 5.0, 6.0, 7.0,
    8.0, 9.0, 7.5, 6.5, 5.0, 6.0, 7.0, 8.0, 9.0, 7.5
)

# dm_goal = (
#     0.05, 0.10, 0.12, 0.15, 0.12, 0.10, 0.05, 0.12, 0.05, 0.10,
#     0.12, 0.15, 0.12, 0.10, 0.05, 0.10, 0.12, 0.15, 0.12, 0.10
# )  # DTLZ2 and DTLZ4

# dm_goal = (
#     0.5, 0.2, 0.8, 4.2, 1.01, 2.06, 7.0, 5.0, 3.0, 2.0,
#     2.042, 4.01, 6.7, 8.08, 10.05, 11.06, 12.07, 8.88, 14.5, 17.0
# )  # WFG4

dm_goal = (
    0.05, 0.03, 0.02, 0.042, 0.01, 0.06, 0.08, 0.05, 0.03, 0.02,
    1.042, 1.01, 1.06, 1.08, 1.05, 2.042, 4.01, 6.06, 8.08, 10.05
)  # WFG9



# ---------------------------------------------------------------------------
# Optimisation setup
# Activate the bounds and constraint corresponding to the test problem.
# ---------------------------------------------------------------------------

# DTLZ2 and DTLZ4
# bounds = np.array([[0, 1] for _ in range(num_objectives)])

# WFG4 and WFG9
bounds = np.asarray([
    [0, 2 * (i + 1)] for i in range(num_objectives)
])

initial_point = np.ones(num_objectives)  # WFG4 and WFG9

weighted_distance = lambda z: sum(
    dm_goal_weights[j] * max(0, z[j] - dm_goal[j]) ** 2
    for j in range(num_objectives)
)

# DTLZ2 and DTLZ4: M-sphere constraint
# constraint_function = lambda z: sum(
#     z[i] ** 2 for i in range(num_objectives)
# )

# WFG4 and WFG9: multiaxial ellipsoid constraint
constraint_function = lambda z: sum(
    (z[i] ** 2) / ((2 * (i + 1)) ** 2)
    for i in range(num_objectives)
)

nonlinear_constraint = NonlinearConstraint(
    constraint_function, 1, 1
)


result = minimize(
    weighted_distance,
    initial_point,
    method="SLSQP",
    bounds=bounds,
    constraints=nonlinear_constraint,
)

print(result)

# DTLZ2 and DTLZ4
# print(
#     "The sum of squared objective values:",
#     sum(result.x[i] ** 2 for i in range(num_objectives)),
# )

# WFG4 and WFG9
print(
    "The sum of squared objective values:",
    sum(
        (result.x[i] ** 2) / ((2 * (i + 1)) ** 2)
        for i in range(num_objectives)
    ),
)

# Note: The solution of this optimisation problem is not applicable to DTLZ2
# because it is not Pareto optimal.