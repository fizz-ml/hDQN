import numpy as np
import random

class ReplayBuffer:
    """ Buffer for storing values over timesteps.
    """
    def __init__(self):
        """ Initializes the buffer.
        """
        pass

    def batch_sample(self, batch_size):
        """ Randomly sample a batch of values from the buffer.
        """
        raise NotImplementedError

    def put(self, *value):
        """ Put values into the replay buffer.
        """
        raise NotImplementedError

class ExperienceReplay(ReplayBuffer):
    """
    Experience Replay stores action, state, reward, terminal signal and action duration
    for each time step.
    """
    def __init__(self, state_size, action_size, capacity):
        """ Creates an Experience Replay of certain capacity.
            Acts like a circular buffer.
        Args:
            state_size:     The size of the state to be stored.
            action_size:    The size of the action to be stored.
            capacity:       The capacity of the experience replay buffer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.length  = 0
        self.capacity = capacity

        self.actions = np.empty((self.capacity, self.action_size), dtype = np.float16)
        self.states = np.empty((self.capacity, self.state_size), dtype = np.float16)
        self.rewards = np.empty(self.capacity, dtype = np.float16)
        self.dones = np.empty(self.capacity, dtype = np.bool)
        self.act_dur = np.empty(self.capacity, dtype = np.float16)

        self.current_index = 0
        self.staged = False

    def batch_sample(self, batch_size):
        """ Sample a batch of experiences from the replay.
        Args:
            batch_size: The number of batches to select
        Returns:
            s_t
            a_t
            r_t
            s_t1
            done
        """
        if batch_size > self.length-3:
            # we might not have enough experience
            raise IOError('batch_size out of range')

        idxs = []
        while len(idxs) < batch_size:
            while True:
                # keep trying random indices
                idx = random.randint(1, self.length - 1)
                # don't want to grab current index since it wraps
                if not( idx == self.current_index and idx == self.current_index - 1 ):
                    idxs.append(idx)
                    break

        s_t = self.states[idxs]
        s_t1 = self.states[[(x+1) for x in idxs]]
        a_t = self.actions[idxs]
        r_t = np.expand_dims(self.rewards[idxs], axis = 1)
        done = self.dones[idxs]

        '''
        j = 0
        print(s_t[j], s_t1[j], a_t[j], r_t[j], done[j])
        j = 1
        print(s_t[j], s_t1[j], a_t[j], r_t[j], done[j])
        raw_input("Press Enter to continue...")
        '''
        return s_t, a_t, r_t, s_t1, done

    def _put(self, s_t, a_t, reward, done):
        self.actions[self.current_index] = a_t
        self.states[self.current_index] = s_t
        self.rewards[self.current_index] = reward
        self.dones[self.current_index] = done
        self._icrement_index()

    def put_act(self, s_t, a_t):
        """ Puts the current state and the action taking into Experience Replay.
        Args:
            s_t:        Current state.
            a_t:        Action taking at this state.
        Raises:
            IOError:    If trying to overwrite previously staged action and state.
        """
        if not self.staged:
            self.actions[self.current_index] = a_t
            self.states[self.current_index] = s_t
            # stage to prevent double staging
            self.staged = True
        else:
            # already staged an action and state
            raise IOError('Trying to override previously staged action and state.')

    def put_rew(self, reward, done):
        """ Completes a staged insertion by adding reward and
            terminal signal to Experience Replay
        Args:
            reward:     Reward received in this step.
            done:       Bool signalling terminal step.
        Raises:
            IOError:    If trying to complete insertion without having staged first.
        """
        if(self.staged):
            self.rewards[self.current_index] = reward
            self.dones[self.current_index] = done
            # unstage and increment index
            self.staged = False
            self._increment_index()
        else:
            # not yet staged state and action
            raise IOError(  'Trying to complete unstaged insertion. Must insert action and state first.')

    def unstage(self):
        """ Unstages any currently staged insertion
        """
        if(self.staged):
            # stage to prevent double staging
            self.staged = False
            self.actions[self.current_index] = None
            self.states[self.current_index] = None

    def _increment_index(self):
        self.current_index = (self.current_index + 1) % self.capacity
        self.length = min(self.capacity-1, self.length + 1)
