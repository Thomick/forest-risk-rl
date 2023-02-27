This folder contains a selection of papers that I directly used for the project and the associated code (I try to keep the list short by adding only the one currently relevant).

# Couture2016
Used to make a simple forest model as an MDP. Consider the whole forest made of multiple plots. The full state is a vector with the state (age class) of each plot. The action is either we harvest the trees of the stand plot or we don't do anything. The risk comes from windthrow that destroy the plots affected. The probability of the destruction event can either be independent on the state of neighboring trees or "local" if it changes as a function of the states of neighboring stand plots.

The simplified toy version I implemented is defined by only 4 variables:
- the number of neighboring plots
- the number of age classes
- the probability
- the base probability of a windthrow event

In addition, the agent can only choose the action for one plot, the reward is only received when gathering the timber (proportional to the age class of the trees) and no reward in case of windthrow.

The main problem of this model is that the state space may become too large. The number of states is $n^k$ where $n$ is the number of age classes and $k$ is the number of plots. This does not allow yet to get an interesting behavior for now. I need to try another implementation that does not require that much states.


# Fei2020
The authors propose algorithms derived from LSVI and Q-Learning based on alternative value functions that consider an exponential utility function as an objective (the value functions are not linear contrary to the risk-neutral setting)