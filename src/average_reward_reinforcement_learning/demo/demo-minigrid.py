
import learners.Generic.Random as lr
from gym_minigrid.wrappers import *

names = [
'MiniGrid-Empty-8x8-v0',
'MiniGrid-FourRooms-v0',
    'MiniGrid-DoorKey-6x6-v0',
'MiniGrid-MultiRoom-N4-S5-v0',
'MiniGrid-Fetch-6x6-N2-v0',
'MiniGrid-GoToDoor-5x5-v0',
'MiniGrid-MemoryS17Random-v0',
'MiniGrid-KeyCorridorS4R3-v0',
'MiniGrid-ObstructedMaze-2Dlh-v0',
'MiniGrid-LavaGapS5-v0',
'MiniGrid-Dynamic-Obstacles-5x5-v0']


env = gym.make(names[np.random.randint(11)])
learner = lr.Random(env)


def animate(env, learner, timeHorizon, rendermode='networkx'):
    observation = env.reset()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    print("New initialization of ", learner.name())
    print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        env.render()
        action = learner.play(state)  # Get action
        observation, reward, done, info = env.step(action)
        #print("S:"+str(state)+" A:"+str(action) + " R:"+str(reward)+" S:"+str(observation) +" done:"+str(done) +"\n")
        learner.update(state, action, reward, observation)  # Update learners
        cumreward += reward
        cumrewards.append(cumreward)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

animate(env,learner,100)

