## State of the project
The project is currently in a state where it can be used as a library for the different forest environments (by importing env.discrete_env and env.linear_dynamic_env). However, the code is not fully documented. 
The experiments are mainly for testing purpose since they have to be modified directly in the code to set up the parameters.

## Code
The code is contained in the forest_risk_rl folder. 

- The folders named forest_risk_rl/envs and forest_risk_rl/learners contain code from Average Reward Reinforcement Learning [repo](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning). 
- forest_risk_rl/basic_model contains the preliminary code for the basic discrete model. 
- forest_risk_rl/envs contains the new forest environments and discrete MDP environments from [Average Reward Reinforcement Learning](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning)
- forest_risk_rl/learners contains the generic learners from [Average Reward Reinforcement Learning](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning)
- forest_risk_rl/experiments contains the experiments and the code for the plots.

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
