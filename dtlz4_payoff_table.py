"""
Calculate an objective minimum for constructing the DTLZ4 payoff table.

Author: Sarah Saeid Pour
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
# num_objectives = 3
# num_xm_variables = 13
# num_variables = num_objectives + num_xm_variables - 1


# ---------------------------------------------------------------------------
# 5 objectives & 50 decision variables
# ---------------------------------------------------------------------------
# num_objectives = 5
# num_xm_variables = 46
# num_variables = num_objectives + num_xm_variables - 1


# ---------------------------------------------------------------------------
# 7 objectives
# ---------------------------------------------------------------------------
# num_objectives = 7
# num_xm_variables = 8
# num_variables = num_objectives + num_xm_variables - 1


# ---------------------------------------------------------------------------
# 9 objectives
# ---------------------------------------------------------------------------
num_objectives = 9
num_xm_variables = 10
num_variables = num_objectives + num_xm_variables - 1


# ---------------------------------------------------------------------------
# 15 objectives
# ---------------------------------------------------------------------------
# num_objectives = 15
# num_xm_variables = 16
# num_variables = num_objectives + num_xm_variables - 1


# ---------------------------------------------------------------------------
# 20 objectives
# ---------------------------------------------------------------------------
# num_objectives = 20
# num_xm_variables = 21
# num_variables = num_objectives + num_xm_variables - 1



bounds = np.array([[0, 1] for _ in range(num_variables)])

# Alternative bounds restricted to the DTLZ4 Pareto front (X_M = 0.5)
# bounds = np.concatenate(
#     (
#         np.array([[0, 1] for _ in range(num_objectives - 1)]),
#         np.array([[0.5, 0.5] for _ in range(num_xm_variables)]),
#     ),
#     axis=0,
# )


# ---------------------------------------------------------------------------
# DTLZ4 objective evaluation
# ---------------------------------------------------------------------------
def evaluate_dtlz4_objective(x):
    """Evaluate the first objective of the DTLZ4 problem."""
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

    return objective_values[0]


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
result = differential_evolution(evaluate_dtlz4_objective, bounds)

print(f"Decision vector: {result.x}")
print(f"Minimum objective value: {result.fun}")

# ---------------------------------------------------------------------------
# Pseudo payoff table
# ---------------------------------------------------------------------------
pseudo_payoff_table = np.arange(num_objectives)
pseudo_payoff_table = np.repeat(
    pseudo_payoff_table, num_objectives
)
pseudo_payoff_table = np.reshape(
    pseudo_payoff_table, (num_objectives, num_objectives)
)

for i in range(num_objectives):
    pseudo_payoff_table[:, i] = np.roll(
        pseudo_payoff_table[:, i], i
    )

# Scale the pseudo payoff table so that the non-diagonal elements in each
# column are uniformly distributed between the ideal (0) and nadir (1).
pseudo_payoff_table = (
    pseudo_payoff_table / (num_objectives - 1)
)

print("\nPseudo payoff table:")
print(pseudo_payoff_table)