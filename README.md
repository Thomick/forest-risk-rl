## State of the project
The project is currently in a state where it can be used as a library for the different forest environments (by importing env.discrete_env and env.linear_dynamic_env). However, the code is not fully documented. 
The experiments are mainly for testing purpose since they have to be modified directly in the code to set up the parameters.

## Code
The code is contained in the src folder. 

- The folders named src/envs and src/learners contain code from Average Reward Reinforcement Learning [repo](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning). 
- src/basic_model contains the preliminary code for the basic discrete model. 
- src/envs contains the new forest environments and discrete MDP environments from [Average Reward Reinforcement Learning](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning)
- src/learners contains the generic learners from [Average Reward Reinforcement Learning](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning)
- src/experiments contains the experiments and the code for the plots.

## Installing and running the experiments
To run the experiments:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python forest_risk_rl/experiments/[experiment_name].py
```
or if you think you already have the necessary packages installed:
```
python -m forest_risk_rl.experiments.[experiment_name]
```


## Resource folder
The resource folder contains some papers that may be useful during discussions.
