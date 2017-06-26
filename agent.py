class Agent:
    """The generic interface for an agent.

        Attributes:
            auxillary_rewards: The list of enabled
            auxillary rewards for this agent
    """
    def train():
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        raise NotImplementedError

    def get_next_action(cur_state,
            prev_reward,
            is_done=False,
            agent_id=None,
            is_test=True):
        """Get the next action from the agent.
            
            Takes a state,reward and possibly auxiliary reward
            tuple and returns the next action from the agent.
            The agent may cache the reward and state 

            Args:
                cur_state: The current state of the enviroment
                prev_reward: The previous reward from the enviroment
                is_done: Signals if a given episode is done.
                is_test: Check to see if the agent is done
                agent_id=None
            Returns:
                The next action that the agent with the given 
                agent_id will carry out given the current state
        """
        raise NotImplementedError

    def save_models(location=None):
        """Save the model to a given location

            Args:
                Location: where to save the model
            Returns:
                None
        """
        raise NotImplementedError
        
    def load_models(location=None):
        """Loads the models from a given location

            Args:
                Location: from where to load the model
            Returns:
                None
        """
        raise NotImplementedError
