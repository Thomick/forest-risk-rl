This folder contains a selection of papers that may be used as reference during discussions.

# Couture2016
Used to make a simple forest model as an MDP. Consider the whole forest made of multiple plots. The full state is a vector with the state (age class) of each plot. The action is either we harvest the trees of the stand plot or we don't do anything. The risk comes from windthrow that destroy the plots affected. The probability of the destruction event can either be independent on the state of neighboring trees or "local" if it changes as a function of the states of neighboring stand plots.

The simplified toy version I implemented is defined by only 4 variables:
- the number of neighboring plots
- the number of age classes
- the probability
- the base probability of a windthrow event

In addition, the agent can only choose the action for one plot, the reward is only received when gathering the timber (proportional to the age class of the trees) and no reward in case of windthrow.

The main problem of this model is that the state space may become too large. The number of states is $n^k$ where $n$ is the number of age classes and $k$ is the number of plots. This does not allow yet to get an interesting behavior for now. I need to try another implementation that does not require that much states.

# Loisel2014
A more realistic model of forest growth "Average forest growth model". Dynamical system with two variables: density of the stand and average basal area (optionally the price of lumber which depend on the time since the last storm). It includes thinning incomes which allows smoothing profits over time. Storms follow a Poisson process. However, the risk is independent of the states of neighboring plots. Allows to anticipate the constraints of the dynamical system for the forest model.
Also features the Faustmann value which corresponds to the discounted value of cutting incomes minus the cost of replanting.

# Couture2021
Multi-objective forest management. Defines reward functions for the different ecosystem services such that timber production, biodiversity and carbon sequestration. 

# Gravell2021
Algorithm to learn LQR with multiplicative noise (with code).
