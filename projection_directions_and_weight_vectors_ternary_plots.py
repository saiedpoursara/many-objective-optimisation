"""
Generate and visualise projection directions and corresponding weight vectors
using ternary plots.

Author: Sarah Saeid Pour
"""

import math

import numpy as np
import pandas as pd
import ternary


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
num_objectives = 3
num_projection_directions = 7

# ---------------------------------------------------------------------------
# Generate candidate projection directions
# ---------------------------------------------------------------------------
# Generate a large uniform random sample.
uniform_sample = np.random.uniform(
    0, 1, size=(20 * num_projection_directions, num_objectives)
)

# Transform the uniform sample using the inverse CDF of the reciprocal
# distribution to generate candidate projection directions.
rud_sample = 1 / (1 - uniform_sample)

sample_sums = rud_sample.sum(axis=1)
rud_sample = rud_sample / sample_sums[:, np.newaxis]

sample_labels = [
    f"s{i + 1}" for i in range(num_projection_directions * 20)
]
rud_sample_df = pd.DataFrame(rud_sample, index=sample_labels)


# ---------------------------------------------------------------------------
# Filter the candidate projection directions
# ---------------------------------------------------------------------------
# Calculate the pairwise Euclidean distances between the sampled directions.
distances = np.full(
    (
        num_projection_directions * 20,
        num_projection_directions * 20,
    ),
    2,
    dtype=float,
)
distances_df = pd.DataFrame(
    distances,
    index=sample_labels,
    columns=sample_labels,
)

for i in range(num_projection_directions * 20):
    for j in range(i, num_projection_directions * 20):
        distance = 0
        for k in range(num_objectives):
            distance += (rud_sample[i, k] - rud_sample[j, k]) ** 2
        distances_df.iloc[i, j] = math.sqrt(distance)

for i in range(num_projection_directions * 20):
    for j in range(num_projection_directions * 20):
        if i < j:
            distances_df.iloc[j, i] = distances_df.iloc[i, j]

for i in range(num_projection_directions * 20):
    distances_df.iloc[i, i] = 2

# Iteratively eliminate one member of the closest pair until the required
# number of projection directions remains.
remaining_labels = sample_labels.copy()

for _ in range(num_projection_directions * 19):
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

final_rud_sample = pd.DataFrame(
    rud_sample_df, index=remaining_labels
)
projection_directions = pd.DataFrame(final_rud_sample).to_numpy()

# ---------------------------------------------------------------------------
# Convert projection directions to weight vectors
# ---------------------------------------------------------------------------
# Weight vector in the ASF: inverse of the projection direction.
weight_vectors = 1 / projection_directions

weight_vector_sums = weight_vectors.sum(axis=1)

# Normalise each weight vector so that its components sum to one.
weight_vectors = weight_vectors / weight_vector_sums[:, np.newaxis]

# ---------------------------------------------------------------------------
# Ternary plots
# ---------------------------------------------------------------------------

# Large normalised uniform sample
print("\nA large sample from the uniform distribution.")

uniform_sample_sums = uniform_sample.sum(axis=1)
normalised_uniform_sample = (
    uniform_sample / uniform_sample_sums[:, np.newaxis]
)
normalised_uniform_sample = pd.DataFrame(normalised_uniform_sample)

figure, tax = ternary.figure()
tax.boundary(linewidth=1.0)
tax.gridlines(multiple=0.1, color="blue")
tax.scatter(
    normalised_uniform_sample[[0, 1]].values,
    marker="o",
    color="red",
    label="Red Points",
)
tax.ticks(axis="brl", linewidth=1, multiple=1)
tax.show()


# Large normalised reciprocal-uniform sample
print("\nA large sample from the reciprocal uniform distribution.")

figure, tax = ternary.figure()
tax.boundary(linewidth=1.0)
tax.gridlines(multiple=0.1, color="blue")
tax.scatter(
    rud_sample_df[[0, 1]].values,
    marker="o",
    color="red",
    label="Red Points",
)
tax.ticks(axis="brl", linewidth=1, multiple=1)
tax.show()


# Final reciprocal-uniform sample: projection directions
print(
    "\nFinal sample from the reciprocal uniform distribution "
    "(projection directions)."
)

figure, tax = ternary.figure()
tax.boundary(linewidth=1.0)
tax.gridlines(multiple=0.1, color="blue")
tax.scatter(
    final_rud_sample[[0, 1]].values,
    marker="o",
    color="red",
    label="Red Points",
)
tax.ticks(axis="brl", linewidth=1, multiple=1)
tax.show()


# Weight vectors used in the ASF: normalised reciprocals of projection directions
print(
    "\nWeight vectors used in the ASF "
    "(normalised reciprocals of projection directions)."
)

weight_vectors_df = pd.DataFrame(weight_vectors)

figure, tax = ternary.figure()
tax.boundary(linewidth=1.0)
tax.gridlines(multiple=0.1, color="blue")
tax.scatter(
    weight_vectors_df[[0, 1]].values,
    marker="o",
    color="blue",
    label="Blue Points",
)
tax.ticks(axis="brl", linewidth=1, multiple=1)
tax.show()