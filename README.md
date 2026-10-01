# Interactive Many-Objective Optimisation

Selected Python implementations developed as part of my PhD research on
interactive decision support for many-objective optimisation problems at the
University of Cape Town.

The research develops interactive methods that help a decision maker explore
the vast trade-off space in many-objective optimisation problems and
progressively focus the search on a region that better reflects their
preferences. The methods use preference information to guide the search while
presenting the decision maker with a cognitively manageable shortlist of
7 ± 2 Pareto-optimal solutions at each interaction, supporting progressive
learning and preference refinement until a final preferred solution is reached.

This repository presents selected implementations from the research. It focuses
on one of the proposed methods and selected instances of the DTLZ4 benchmark
problem, together with supporting optimisation, sampling, and visualisation
code. As the associated research is currently being prepared for publication,
the complete implementation is not yet publicly available. Additional code
will be added following publication.

The main implementation included here demonstrates the geometric
multiple-reference-point method on DTLZ4 instances with different numbers of
objectives. The remaining scripts support components of the method,
experimental setup, performance assessment, and visualisation.

## Repository Contents

### Interactive geometric method using multiple reference points

- `geometric_method_multiple_rps_dtlz4.py`  
  Implements an interactive geometric method based on multiple reference
  points, demonstrated on DTLZ4 instances with 3, 5, 7, 9, 15, and 20
  objectives.

### Optimisation and performance assessment

- `closest_point_to_pf_from_dm_goal_dtlz4.py`  
  Calculates the point on the DTLZ4 Pareto front closest to a simulated
  decision maker goal using differential evolution, across instances with
  different numbers of objectives.

- `dtlz4_closest_point_to_dm_goal_ga.py`  
  Uses a binary-coded genetic algorithm to search for a DTLZ4 Pareto-optimal
  solution closest to a simulated decision maker goal.

- `dtlz4_payoff_table.py`  
  Generates the pseudo-payoff table used in the DTLZ4 experiments.

- `minimum_weighted_distance_to_goal.py`  
  Calculates the minimum weighted distance from a decision maker goal to
  hyperspherical and multiaxial ellipsoidal surfaces.

### Sampling and visualisation

- `projection_directions_and_weight_vectors_ternary_plots.py`  
  Generates projection directions associated with the reference points and
  the corresponding weight vectors used in the achievement scalarising
  function, and visualises them using ternary plots.

- `uniform_points_on_hypersphere.py`  
  Generates uniformly distributed points on a hypersphere and applies an
  iterative filtering procedure to select a more dispersed subset.

## Requirements

Developed and tested with Python 3.9.

Depending on the script, the following Python packages are used:

- NumPy
- Pandas
- SciPy
- Matplotlib
- python-ternary

## Usage

The scripts can be run independently. Where applicable, configurations for
different numbers of objectives are provided near the beginning of each script
and can be activated as required.

## Notes

Decision maker preferences used in the numerical examples are simulated for
experimental purposes.

## Author

**Sarah Saeid Pour**  
PhD in Statistical Sciences, 2026  
Department of Statistical Sciences  
University of Cape Town

PhD thesis: *Reference Point Methodology for Interactive Exploration of the Pareto Frontier*