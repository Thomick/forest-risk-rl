from rlberry.agents import AgentWithSimplePolicy
from learners.discreteMDPs.OptimalControl import Opti_controller
import numpy as np
from forest_risk_rl.risk_measures import compute_empirical_cvar, compute_group_risk


class ARRLAgent(AgentWithSimplePolicy):
    def __init__(
        self,
        env,
        learner_ctor,
        learner_args={},
        render=False,
        track_group_risk=False,
        **kwargs
    ):
        self.track_group_risk = track_group_risk
        self.render = render
        self.env = env[0](**env[1])
        self.learner = learner_ctor(self.env.nS, self.env.nA, **learner_args)
        self.name = self.learner.name()
        super().__init__(env, **kwargs)
        self.global_step = 0

    def fit(self, budget, **kwargs):
        observation = self.env.reset()
        self.learner.reset(observation)
        episode_means = 0.0
        episode_rewards = 0.0
        for t in range(budget):
            self.global_step += 1
            state = observation
            action = self.policy(state)  # Get action
            observation, reward, done, info = self.env.step(action)
            self.update_learner(state, action, reward, observation)  # Update learners
            # print("info:",info, "reward:", reward)
            episode_rewards += reward
            try:
                episode_means += info["mean"]
            except TypeError:
                episode_means += reward
            total_episodes = 0
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                observation = self.env.reset()
                total_episodes += 1
                if self.writer is not None:
                    self.writer.add_scalar(
                        "episode_rewards", episode_rewards, self.global_step
                    )
                    self.writer.add_scalar(
                        "total_episodes", total_episodes, self.global_step
                    )
                    self.writer.add_scalar(
                        "episode_means", episode_means, self.global_step
                    )
                else:
                    print("No writer")
                episode_rewards = 0.0
                episode_means = 0.0
            if self.writer is not None:
                self.writer.add_scalar("rewards", reward, self.global_step)
                if self.track_group_risk:
                    self.writer.add_scalar(
                        "group_risk",
                        compute_group_risk(
                            self.env.index_to_state(observation),
                            self.env.group_risk_weights,
                        ),
                        self.global_step,
                    )

            if self.render:
                self.env.render()

    def policy(self, observation):
        return self.learner.play(observation)

    def update_learner(self, state, action, reward, next_state):
        self.learner.update(state, action, reward, next_state)

    def eval(self, metric="reward", **kwargs):
        if metric == "reward":
            return super().eval(**kwargs)
        elif metric == "cvar":
            return self.eval_cvar(**kwargs)
        elif metric == "group_risk":
            return self.eval_group_risk(**kwargs)

    def eval_cvar(self, eval_horizon=10**5, n_simulations=10, gamma=1.0, alpha=0.05):
        reward_samples = []
        for sim in range(n_simulations):
            observation = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation)
                observation, reward, done, _ = self.eval_env.step(action)
                reward_samples.append(reward)
                tt += 1
                if done:
                    break
        return compute_empirical_cvar(reward_samples, alpha)

    def eval_group_risk(self, eval_horizon=10**5, n_simulations=10):
        grouprisk_sample = []
        for sim in range(n_simulations):
            observation = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation)
                observation, reward, done, _ = self.eval_env.step(action)
                grouprisk_sample.append(
                    compute_group_risk(
                        self.env.index_to_state(observation),
                        self.env.group_risk_weights,
                    )
                )
                tt += 1
                if done:
                    break
        return np.mean(grouprisk_sample)


class OptAgent(AgentWithSimplePolicy):
    name = "Optimal Agent"

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.opti_controller = Opti_controller(
            self.env, self.env.observation_space.n, self.env.action_space.n
        )
        self.global_step = 0

    def fit(self, budget, **kwargs):
        observation = self.env.reset()
        self.opti_controller.reset(observation)
        for t in range(budget):
            self.global_step += 1
            state = observation
            action = self.policy(state)  # Get action
            observation, reward, done, info = self.env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                observation = self.env.reset()
            if self.writer is not None:
                self.writer.add_scalar("rewards", reward, self.global_step)
                try:
                    self.writer.add_scalar("means", info["mean"], self.global_step)
                except TypeError:
                    self.writer.add_scalar("means", reward, self.global_step)

    def policy(self, observation):
        return self.opti_controller.play(observation)


class Random:
    def __init__(self, nS, nA, name="Random Agent"):
        self.nS = nS
        self.nA = nA
        self.agentname = name

    def name(self):
        return self.agentname

    def reset(self, inistate):
        ()

    def play(self, state):
        return np.random.randint(self.nA)

    def update(self, state, action, reward, observation):
        ()
