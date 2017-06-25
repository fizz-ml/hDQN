import torch
import torch.optim as opt
from torch.autograd import Variable
from torch import FloatTensor as FT
import agent
from replay_buffer import ExperienceReplay
import numpy as np
import dill
from torch.utils.serialization import load_lua
import model_defs.ddpg_models.mountain_cart.critic as critic
import model_defs.ddpg_models.mountain_cart.actor as actor
import random

#Default hyperparameter values
REPLAY_BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 1
LEARNING_RATE_CRITIC = 0.01
LEARNING_RATE_ACTOR = 0.01
ACTOR_ITER_COUNT = 1000
CRITIC_ITER_COUNT = 1000
BATCH_SIZE = 100
EPSILON = 0.01

class DQNController:
    """
    Attributes:
           replay_buffer: The DDPGAgent replay buffer
    """

    """
    @property
    def actor(self):
        return self.actor

    @property
    def critic(self):
        return self.critic

    @property
    def replay_buffer(self):
        return self.replay_buffer
    """

    def __init__(self,
            subroutines,
            critic_path,
            state_size = 1,
            action_size = 1,
            buffer_size = REPLAY_BUFFER_SIZE,
            gamma = DISCOUNT_FACTOR,
            alpha = LEARNING_RATE_CRITIC,
            iter_count = CRITIC_ITER_COUNT,
            batch_size = BATCH_SIZE,
            ):
        """Constructor for the DDPG_agent

        Args:
            actor_path: location of the actor_t7

            critic_path: location of the critic_t7

            buffer_size: size of the replay buffer

            alpha: The learning rate

            gamma: The discount factor

        Returns:
            A DDPGAgent object
        """
        super(DDPGAgent, self).__init__(auxiliary_losses)
        #state_size + 1 because we append reward to the input
        state_size = state_size + 1

        #Initialize experience replay buffer
        self.replay_buffer = ExperienceReplay(state_size, action_size, buffer_size)
        #TODO

        self._caller = caller

        #initialize parameters
        self.epsilon = 0.35
        self._alpha = alpha
        self.iter_count = iter_count
        self._gamma = gamma
        self._batch_size = batch_size
        self._state_size = state_size
        self._action_size = action_size

        #Specify model locations
        self._critic_path = critic_path

        #initialize models
        self.load_models()

        #Initialize optimizers
        self._critic_optimizer = opt.Adam(self.critic.parameters(), lr=self._critic_alpha)


    def train(self, q_caller = None):
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        self.epsilon = self.epsilon * 0.99992
        #update_critic
        for i in range(self._critic_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            done = upcast(done)
            s_t = upcast(s_t)
            a_t = upcast(a_t)
            s_t1 = upcast(s_t1)
            r_t = upcast(r_t)
            a_t1, _ = self.actor.forward(s_t1,[])
            critic_target = r_t + self._gamma*(1-done)*self._target_critic.forward(s_t1,a_t1)
            td_error = (self.critic.forward(s_t,a_t)-critic_target)**2

            #preform one optimization update
            self._critic_optimizer.zero_grad()
            mean_td_error = torch.mean(td_error)
            mean_td_error.backward()
            self._critic_optimizer.step()


        #update_actor
        for i in range(self._actor_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            done = upcast(done)
            s_t = upcast(s_t)
            a_t = upcast(a_t)
            s_t1 = upcast(s_t1)
            r_t = upcast(r_t)
            a_t1,aux_actions = self.actor.forward(s_t1,self.auxiliary_losses.keys())
            expected_reward = self.critic.forward(s_t1,a_t1)

            total_loss = -1*expected_reward
            for key,aux_reward_tuple in self.auxiliary_losses.items():
                aux_weight,aux_module = aux_reward_tuple
                total_loss += aux_weight*aux_module(aux_actions[key],s_t,a_t,r_t,s_t1,a_t1)

            mean_loss = torch.mean(total_loss)

            #print('LOSS:', mean_loss, 'Eps', self.epsilon)
            #preform one optimization update
            self._actor_optimizer.zero_grad()
            mean_loss.backward()
            self._actor_optimizer.step()

        # TODO: Freeze less often
        self._target_critic.load_state_dict(self.critic.state_dict())



    def act(self,
            env,
            is_test=False):
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
        tot_r = 0
        dur = 0
        while True:
            cur_action = None
            cur_state = env.cur_obs
            if is_test:
                a, _ = self.actor.forward(upcast(np.expand_dims(cur_state,axis=0)),[])
                cur_action = a.data.cpu().numpy()
            elif random.random() < self.epsilon:
                cur_action = np.expand_dims(np.random.randn(self._action_size),axis=0)
            else:
                a, _ = self.actor.forward(upcast(np.expand_dims(cur_state,axis=0)),[])
                cur_action = a.data.cpu().numpy()


            self.replay_buffer.put(cur_state, cur_action, r, done)

            tot_r += r
            dur += 1

        return tot_r, dur, done

    def execute_action(self, env, a):
        # A real actual action
        if a < action_size:
            _, r, done = env.next_obs(a)
            dur = 1

        # A subroutine call
        else:
            sub_idx = a - action_size
            r, dur, done = subcontrollers[sub_idx].act(env)

        return r, dur, done

    def save_models(self, locations=None):
        """Save the model to a given locations

            Args:
                Locations: where to save the model
            Returns:
                None
        """
        #Return all weights and buffers to the cpu
        self.actor.cpu()
        self.critic.cpu()

        #Save both models
        actor_file=open(self._actor_path,"wb")
        dill.dump(self.actor,actor_file)
        critic_file=open(self._critic_path,"wb")
        dill.dump(self.critic,critic_file)

    def load_models(self, locations=None):
        # TODO: Make it actually do what it says
        #TODO: Remove hard coding of data
        """Loads the models from given locations

            Args:
                Locations: from where to load the model
            Returns:
                None
        """
        actor_file=open(self._actor_path,"rb")
        self.actor = actor.Actor(self._state_size,self._action_size) #dill.load(actor_file)
        critic_file=open(self._critic_path,"rb")
        self.critic = critic.Critic(self._state_size + self._action_size, 1)#dill.load(critic_file)
        self._target_critic = critic.Critic(self._state_size + self._action_size,1)#dill.load(critic_file)

        #Move weights and bufffers to the gpu if possible
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
            self._target_critic()

def upcast(x):
    ''' Upcasts x to a torch Variable.
    '''
    #TODO: Where does this go?
    return Variable(FT(x.astype(np.float32)))

