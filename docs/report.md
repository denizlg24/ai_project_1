# Overall comparisons

Running all the algorithms with different parameters but always from the same starting solution gave us very good insight.

Of course the rate of change was different and we could still observe different behaviour of the scores across algorithms.

The best algorithm was by far Simulated Annealing. It reached the best overall score and still with low runtime in our tests (`docs/experiment_runs.csv`).

This is due to two main details.

Our implementation of Simulated Annealing only randomly reassigns one family's assigned day, as long as the move is allowed. Because of this, `delta_cost` only recomputes the affected days and is effectively O(1), which makes the program relatively efficient (`src/core/problem.py`).

Secondly, due to the nature of the problem (many local optimums), Simulated Annealing is a good solution because it can accept some worse moves during exploration (`exp(-delta/T)` in `src/algorithms/simulated_annealing.py`) and escape local minima. It does not guarantee the global minimum, but in practice it consistently converged to much better solutions than the other approaches.

One downside of our implementation is that because the reassignment is random and only happens for one family, when we start from an already close-to-optimal solution we usually are not able to improve it significantly. Our hypothesis is that random exploration at high temperature can quickly jump to a worse region, and then time is spent recovering from that region.

Our second best algorithm is Variable Neighbourhood Search. Again due to the nature of the problem and having many local optima, VNS helps us escape those by reshuffling the solution space. It works in a comparable way to Simulated Annealing, but because it does not have a temperature-based acceptance of worse moves, it is still susceptible to getting stuck in a local optimum where none of the neighbourhoods are better (`src/algorithms/vns.py`). Also, because of the larger perturbations and frequent full-score evaluations, it takes longer and is less effective than SA in our implementation.

The neighbourhoods developed were the following 4:

    1. Move a random family to a random preference choice
    2. Swap two random families
    3. Move family A's choice to family B's choice then reassign family B to a new preferred day
    4. Pick a random day, redistribute up to 5 families to other preferred days.

Finally, the algorithm that started as worst performing in solution quality was the Genetic Algorithm. The nature of the problem does not directly translate to a genetic approach. It is hard to detect features that should be passed to the next generation and, while we need population diversity for a good GA, that diversity can also hurt score stability because small assignment changes can create large score changes.

Our final conclusion is still that SA is the best fit for this problem in our current codebase, VNS is a valid second option, and GA is now improved but still not competitive enough without deeper operator/representation improvements.
