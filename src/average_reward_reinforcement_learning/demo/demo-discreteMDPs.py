#from experiments.runExperiments import *
import environments.RegisterEnvironments as bW
import learners.Generic.Random as lr
import learners.discreteMDPs.Human as lh
import learners.discreteMDPs.PSRL as bl



#################################
# Running a single experiment:
#################################

def animate(env, learner, timeHorizon):
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
        # print("S:"+str(state)+" A:"+str(action) + " R:"+str(reward)+" S:"+str(observation) +" done:"+str(done) +"\n")
        learner.update(state, action, reward, observation)  # Update learners
        cumreward += reward
        cumrewards.append(cumreward)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


def demo_riverSwim():
     testName = 'ergo-river-swim-6'
     envName = (bW.registerWorlds[testName])(0)
     env = bW.makeWorld(envName)
     env.env.rendermode = 'networkx'
     learner = lr.Random(env)
     #learner = lh.Human(env)
     # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     #print(env.renderers.keys(),env.rendermode)
     animate(env, learner, 100)
     #animate(env, learner, 100, 'text')
     #animate(env, learner, 100, 'text')


def demo_randomGrid():
     #testName = 'grid-random-88'
     testName = 'grid-2-room'

     envName = (bW.registerWorlds[testName])(0)
     env = bW.makeWorld(envName)
     env.env.rendermode = 'gw-pyplot'
     learner = lr.Random(env)
     #learner = bl.PSRL(env.observation_space.n, env.action_space.n, delta=0.05)
     #learner = lh.Human(env)
     # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     #animate(env, learner, 100, 'text')
     animate(env, learner, 100)

def demo_randomMDP():
    testName = 'random-12'
    envName = (bW.registerWorlds[testName])(0)
    env = bW.makeWorld(envName)
    env.env.rendermode = 'text'
    #env.env.rendermode = 'networkx'
    #env.env.rendermode = 'pydot'
    learner = lr.Random(env)
    # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
    animate(env, learner, 5)
    #animate(env, learner, 50, 'text')
    #
    #
    # testName = 'random100'
    # envName = (bW.registerWorlds[testName])(0)
    # env = bW.makeWorld(envName)
    # learner = lr.Random(env)
    # # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
    # animate(env, learner, 100, 'text')

#demo_randomMDP()
#demo_riverSwim()
demo_randomGrid()