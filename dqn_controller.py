import torch
import torch.optim as opt
from torch.autograd import Variable
from torch import FloatTensor as FT
import agent
from replay_buffer import ExperienceReplay
import numpy as np
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

    def __init__(self, config, env):
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
        #Initialize experience replay buffer
        self.replay_buffer = ExperienceReplay(state_size, action_size, buffer_size)
       
        #tree metadata
        self._parent_ids = config['parent_ids'] #value of -1 indicates tree root  #subcontroller ids
        self._subcontroller_ids = config['subcontroller_ids']
        self._is_root = (self._parent_ids == None)#TODO

        if (subcontroller_ids == None):
            self._true_actor = True
        else:
            self._true_actor = False

        #for idx in subcontroller_ids:
        #    self.subcontrollers = []

        #make sure config spits out these instead of hardcoding them
        #controller parameters
        self.epsilon = 0.35
        self._alpha = config['alpha']
        self.iter_count = config['iter_count']
        self._gamma =config[' gamma']
        self._batch_size = config['batch_size']
        self._state_size = env.obs_size()
               
        if(self._true_actor):
            self._action_size = env.action_size() + 1
        else:
            self._action_size = len(subcontroller_ids) + 1 
        if(self._is_root):
            self._action_size = self._action_size - 1 #if root, no callback action needed
        
        
        #Specify model locations
        #self._critic_path = critic_path
        self._Q = self.create_model(self._state_size, self._action_size) #TODO merge these  config-like params nicely        


        #initialize models
        #self.load_models()

        #Initialize optimizer
        self._Q_optimizer = opt.Adam(self._Q.parameters(), lr=self._alpha)
    
    
    def create_model(self, state_size, action_size):
        
        model = torch.nn.Sequential(
                    torch.nn.Linear(state_size, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, action_size),
                    torch.nn.ReLU()
                )
        return model

    def train(self, q_caller = None):
        """Trains the agent for a bit.

            Args:
                
            Returns:
                None
        """
        self.epsilon = self.epsilon * 0.99992 #TODO prove this is optimal


        #get argmax_a' Q(s,a')
        
        s_t,a_t,r_t,s_t1,done,act_dur= self.replay_buffer.batch_sample(self._batch_size)

        #DQN loss
        loss = torch.nn.MSEloss(self._Q(s_t)[a_t], r_t +  self._gamma*np.max(self._Q.forward(s_t1).detach(),axis=1)) #ADD OPTIMIZER
        self._Q_optimizer.zero_grad()
        loss.backward(self._Q.parameters())
        self._Q_optimizer.step()

        

        #update_critic
        
    def set_controller_tree(controller_tree):
        self._controller_tree = controller_tree
    
    def act(self,
            env,
            caller_id,
            is_test=False):
        # Keep track of total reward and duration
        tot_r = 0
        tot_dur = 0

        while True:
            # Forward step
            cur_state = env.cur_obs
            cur_action = self.choose_action(cur_state, is_test)
            r, dur, done, ret = self.execute_action(env, cur_action)

            # Store the tuple in replay
            self.replay_buffer.put(cur_state, cur_action, r, done, dur, caller_id)

            # Update totals
            tot_r += r
            tot_dur += dur

            # End if episode end or return action called
            if done or ret:
                break

        # Return control to caller
        return tot_r, dur, done

    def choose_action(self, cur_state, is_test=False):
        """Get the next action from the agent.

            Takes a state and returns the next action from the agent.
            The agent may cache the reward and state

            Args:
                cur_state: The current state of the enviroment
                is_test: Check to see if the agent is done
            Returns:
                The next action
        """
        #TODO: ensure Q network just returns Q(s,a) over all a
        cur_action = None
        
        if random.random() < self.epsilon and is_test == False:
            cur_action = np.floor(np.expand_dims(np.random.randn(self._action_size),axis=0)).astype(int)

        else:
            a = np.argmax(self._Q.forward(upcast(np.expand_dims(cur_state,axis=0)),[]))     
            cur_action = a.data.cpu().numpy() #TODO:why this?

        return cur_action

    def execute_action(self, env, a, is_test):
        ret = False
        #TODO: what to do when root controller
        # Return control to caller
        if a == 0 and not self._is_root:
            r = 0
            dur = 0
            done = False
            ret = True

        # Perform real actual action
        elif self._true_actor:
            _, r, done = env.next_obs(a)
            dur = 1

        # Make a subroutine call
        else:
            if(self._is_root == False):
                sub_idx = a - 1 #submodule calls start from action 1 if non root actor
            else:
                sub_idx = a #if root

            r, dur, done = self._controller_tree[subcontrollers[sub_idx]].act(env, self._id, is_test=is_test)

        return r, dur, done, ret

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

