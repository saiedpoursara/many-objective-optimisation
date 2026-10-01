"""
Generate and visualise points sampled on a hypersphere.

Author: Sarah Saeid Pour
"""

import matplotlib.pyplot as plt
import numpy as np


# Sampling algorithm 1: Gaussian normalisation
def get_random_samples_on_n_sphere(n, radius, num_samples):
    """Generate uniformly distributed points on an (n - 1)-sphere."""

    # Return 'num_samples' vectors of dimension n, uniformly distributed
    # on the surface of an (n - 1)-sphere with radius 'radius'.
    # Rationale: https://mathworld.wolfram.com/HyperspherePointPicking.html

    samples = np.random.default_rng().normal(size=(num_samples, n))

    return (
        radius
        / np.sqrt(np.sum(samples**2, axis=1, keepdims=True))
        * samples
    )


def get_most_scattered_samples(n, radius, num_samples):
    """
    Select a maximally dispersed subset of points on an (n - 1)-sphere.

    A larger candidate sample is generated first. The two closest points
    are then identified, and the point lying in the denser local region
    is removed. This process is repeated until the required number of
    points remains.
    """
    candidate_samples = get_random_samples_on_n_sphere(
        n, radius, num_samples * 20
    )

    remaining_samples = candidate_samples.copy()

    while len(remaining_samples) > num_samples:
        # Calculate the Euclidean distances between all pairs of points.
        differences = (
            remaining_samples[:, np.newaxis, :]
            - remaining_samples[np.newaxis, :, :]
        )
        distances = np.linalg.norm(differences, axis=2)

        # Exclude each point's distance from itself.
        np.fill_diagonal(distances, np.inf)

        # Identify the two closest points.
        closest_pair = np.unravel_index(
            np.argmin(distances), distances.shape
        )
        point_1, point_2 = closest_pair

        # Determine the second-closest point to each member of the pair.
        distances_1 = distances[point_1].copy()
        distances_2 = distances[point_2].copy()

        distances_1[point_2] = np.inf
        distances_2[point_1] = np.inf

        second_closest_distance_1 = np.min(distances_1)
        second_closest_distance_2 = np.min(distances_2)

        # Remove the member of the closest pair that lies in the denser
        # local region.
        if second_closest_distance_2 <= second_closest_distance_1:
            remove_index = point_2
        else:
            remove_index = point_1

        remaining_samples = np.delete(
            remaining_samples, remove_index, axis=0
        )

    return remaining_samples


# Generate a test sample using the Gaussian normalisation method.
test_sample = get_random_samples_on_n_sphere(3, 1, 7)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_aspect("auto")

# Draw the unit sphere.
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

ax.plot_wireframe(x, y, z, color="pink")

# Draw the sampled points.
x_sample = test_sample[:, 0]
y_sample = test_sample[:, 1]
z_sample = test_sample[:, 2]

ax.scatter(x_sample, y_sample, z_sample, color="r")

plt.show()
