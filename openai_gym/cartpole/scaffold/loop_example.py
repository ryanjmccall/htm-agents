import gym
from gym import wrappers

def main():
    episodeCount = 20
    stepsPerEpisode = 100

    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, "/tmp/cartpole-experiment-1")
    for episode in range(episodeCount):
        observation = env.reset()
        for t in range(stepsPerEpisode):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == "__main__":
    main()
