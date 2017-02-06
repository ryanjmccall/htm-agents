# HTM Agent for "Cart-Pole-v0" environment from OpenAI Gym
import gym
import numpy as np
import random

from htm_learner import HtmLearner


def run(env, learner, num_episodes, num_steps):
    scores = []
    ave_cumulative_reward = None
    for episode in range(num_episodes):
        observation = env.reset()

        # HTM won't have an initial prediction, so initially choose randomly?
        action = env.action_space.sample()
        state = learner.compute(observation, action)

        cumulative_reward = 0
        for t in range(num_steps):
            env.render()
            action = learner.bestAction(state)
            observation, reward, done, info = env.step(action)

            cumulative_reward += learner.discount * reward
            newState = learner.compute(observation, action)
            learner.update(state, action, newState, reward)
            state = newState

            if done:
                learner.updateWhenDone(cumulative_reward, ave_cumulative_reward)
                learner.reset()

                scores.append(t + 1)
                print "Steps for episode {}: {}".format(episode, t + 1)
                print "Ave steps last 100: {:.2f}".format(
                    np.mean(scores[-100:]))
                print
                break


def main():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    envId = "CartPole-v0"
    env = gym.make(envId)

    alpha = 0.3
    epsilon = 0.75
    epsilonDecay = 0.99
    discount = 0.95
    k = 0.01
    learner = HtmLearner(env, alpha, epsilon, epsilonDecay, discount, k)

    num_episodes = 1000
    num_steps = 200
    run(env, learner, num_episodes, num_steps)


if __name__ == "__main__":
    main()
