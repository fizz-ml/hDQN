from environment import GymEnvironment
#from ddpg_agent import DDPGAgent
import numpy as np
import sys
import random
import os
import json
from dqn_controller import DQNController
'''

The hierarchy is specified by a tree list

Each controller in the tree has:
    a unique id, 
    list of sub-controller ids,
    parent id

action 0 is reserved for callback to parent

    0
   / \
  1   2
 /\   /\
3  4 5  6


eg, controller_tree = [[1,2],[3,4],[5,6], [],[],[],[]]
    controller_network_configs = [config['0'], config['1'], config['2'], ...
                                    config[num_controllers -1]]
    
     

Issues encountered: no guarantee batch_size sample will be available for training each controller    

'''




tree_config = [
    {'subcontroller_ids':[1,2], 'parent_ids':None,'alpha':0.001, 'gamma':0.99, 'iter_count':20, 'batch_size': 16},
    {'subcontroller_ids':None, 'parent_ids':0,'alpha':0.001, 'gamma':0.99 , 'iter_count':20, 'batch_size': 16},
    {'subcontroller_ids':None, 'parent_ids':0, 'alpha':0.001, 'gamma':0.99, 'iter_count':20, 'batch_size':16}

]

global_config = {
    'env': {'name': 'Acrobot-v1' },
    'subroutines':tree_config,
    'train': {'episodes':500}


}

conf = global_config
global_steps = 1000
def main():
    #json_data = open(sys.argv[1]).read()
    #conf = json.loads(json_data)


    run = Runner(conf['env'], conf['subroutines'])
    for i in range(global_steps):
        run.train(conf['train'])

class Runner:
    def __init__(self, env_config, controller_configs):
        self.env = GymEnvironment(name = env_config['name'])
        #self.env.reset()
        self.controllers = [] #controller is stored at index controller_id
        for config in controller_configs:
            c = DQNController(config,self.env)
            self.controllers.append(c)
        for controller in self.controllers:
            controller.set_controller_tree(self.controllers)


    def train(self, train_config):
        self.controllers[0].act(self.env) #TODO: rn root controller should be at index 0

        episodes = train_config['episodes']
        for ep in range(episodes):
            self.controllers[0].act(self.env,0)
        
        for c in self.controllers:
            c.train()
            

if __name__ == "__main__":
    main()
