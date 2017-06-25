import gym
from gym import wrappers
import numpy as np

class Environment:
    """ Defines an environment the actor interacts with.
    """

    def __init__(self):
        """ Initializes an environment.
        """
        pass

    def next_obs(self, cur_action):
        """ Takes an action in the environment.

        """
        raise NotImplementedError

    def new_episode(self):
        """ Starts a new episode of the environment.
        """
        raise NotImplementedError

    @property
    def action_shape(self):
        """ Returns the shape of the action.
        """
        raise NotImplementedError

    @property
    def obs_shape(self):
        """ Returns the shape of the observation.
        """
        raise NotImplementedError


class GymEnvironment(Environment):
    def __init__(self, name, monitor = False):
        """ Initializes a gym environment.
        Args
            monitor:        If True, wraps the environment with the openAI gym monitor.
        """
        self.env = gym.make(name)
        self.env = wrappers.Monitor(self.env, './results/cart_pole_1')
        self._cur_obs = None

    def next_obs(self, cur_action, render = False):
        """ Runs a step in the gym environment.
        Args:
            action:         Current action to perform
            render:         (Optional) Wether to render environment or not.

        Returns:
            obs:            State of the environment after step.
            reward:         Reward received from the step.
            done:           Bool signaling terminal step.
        """
        self.cur_obs, self.cur_reward, self.done, _ = self.env.step(cur_action)
        if (not all(np.isfinite(self.cur_obs))) and (not all(np.isfinite(self.cur_reward))):
            import pdb
            pdb.set_trace()
        if render:
            self.env.render()
        if self.done:
            self.new_episode()
        return self.cur_obs, self.cur_reward, self.done

    def new_episode(self):
        """ Initiates a new episode by resetting the environment.
        Returns:
            obs:    Initial observation of the new episode.
        """
        self.cur_obs = self.env.reset()
        self.env.render()
        return self.cur_obs

    @property
    def action_size(self):
        return self.env.action_space.shape

    @property
    def obs_size(self):
        return self.env.observation_space.shape

    @property
    def cur_obs(self):
        return self._cur_obs

    @cur_obs.setter
    def cur_obs(self, value):
        self._cur_obs = value
