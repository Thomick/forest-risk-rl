## Code
The code is contained in the src folder. 

- The folders named src/envs and src/learners contain code from Average Reward Reinforcement Learning [repo](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning). 
- src/basic_model contains the preliminary code for the basic model. 
- src/ contains the currently developed environments and experiments.

To run the experiments:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python forest_risk_rl/experiments/[experiment_name].py
```
or if you have the necessary packages installed:
```
cd forest_risk_rl
python -m experiments.[experiment_name]
```


## Resource folder
The resource folder contains some papers that may be useful during discussions.
