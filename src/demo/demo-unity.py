
from gym import Env, spaces
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


unitypath='../environments/unity-ml-agents/unity/'

def DiscreteStateDiscreteAction():
    unity_env = UnityEnvironment(unitypath+"/PlantSingleAgent")
    env = UnityToGymWrapper(unity_env,0)


    env.reset()
    for i in range(10):
        states = env.observation_space
        print(states)
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        #print(states)
        #s=states.sample()
        #print("s:", len(s))
        #a = input("action?")
        state,reward,isfinal,info=env.step(a)
        print("S:",state)
        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)

def VectorStateDiscreteAction():
    unity_env = UnityEnvironment(unitypath+"/PushBlockSingleAgent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    paststate=np.zeros(210)
    for i in range(60):
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        states= env.observation_space
        print(states)
        #s=states.sample()
        #print("s:", len(s))
        #a = input("action?")
        state,reward,isfinal,info=env.step(a)
        print("S:",state)
        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)
        deltastates=state-paststate
        print("D:",deltastates)
        paststate = state


def VectorStateVectorAction():
    unity_env = UnityEnvironment(unitypath+"/3DBallSingleAgent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    done = False
    i=1
    while(i<60) and (not done):
        print("I:",i)
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        #a = input("action?")
        state,reward,isfinal,info=env.step(a)
        print("S:",state)
        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)
        done = done or isfinal
        i=i+1


def ImageStateDiscreteAction():
    unity_env = UnityEnvironment(unitypath+"/gridworldsingleagent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    for i in range(20):
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        a = input("action?")
        state,reward,isfinal,info=env.step(a)
        #We get the state as an image from the Unity camera !
        #print(len(state),len(state[0]),len(state[0][0]))
        # If output is image:
        plt.figure(1)
        plt.imshow(state)
        plt.show()
        plt.pause(0.01)

        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)

        #env.render()

#Testing different environments:
################################
#DiscreteStateDiscreteAction()#OK
#ImageStateDiscreteAction()


#VectorStateDiscreteAction()#PushBlockSingleAgent
VectorStateVectorAction()#3DBallSingleAgent
