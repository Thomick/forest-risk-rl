from rlberry.agents import AgentWithSimplePolicy


class ARRLAgent(AgentWithSimplePolicy):
    def __init__(self, env, learner_ctor, learner_args={}, **kwargs):
        super().__init__(env, **kwargs)
        self.learner = learner_ctor(self.env.nS, self.env.nA, **learner_args)
        self.global_step = 0

    def fit(self, budget, **kwargs):
        observation = self.env.reset()
        self.learner.reset(observation)
        # cumrewards = []
        episode_means = 0.0
        episode_rewards = 0.0
        # cummeans = []
        print(
            "[Info] New initialization of ",
            self.learner.name(),
            " for environment ",
            self.env.name,
        )
        # print("Initial state:" + str(observation))
        for t in range(budget):
            self.global_step += 1
            state = observation
            action = self.policy(state)  # Get action
            observation, reward, done, info = self.env.step(action)
            self.learner.update(state, action, reward, observation)  # Update learners
            # print("info:",info, "reward:", reward)
            episode_rewards += reward
            try:
                episode_means += info["mean"]
            except TypeError:
                episode_means += reward
            # cumrewards.append(cumreward)
            # cummeans.append(cummean)
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

            # self.env.render()
        # print("Cumreward: " + str(cumreward))
        # print("Cummean: " + str(cummean))
        # return cummeans #cumrewards,cummeans

    def policy(self, observation):
        return self.learner.play(observation)
