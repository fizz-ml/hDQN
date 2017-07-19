from environment import GymEnvironment
from ddpg_agent import DDPGAgent
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



eg, controller_tree = [[1,2],[3,4],[5,6]]
    controller_network_configs = [config['0'], config['1'], config['2'], ...
                                    config[num_controllers -1]]
    
     

   

'''




def main():
    json_data = open(sys.argv[1]).read()
    conf = json.loads(json_data)


    run = Runner(conf['env'], conf['subroutines'])
    run.train(conf['train'])

class Runner:
    def __init__(self, env_config, controller_configs):
        self.env = GymEnvironment(name = env_config["name"])
        self.controllers = [] #controller is stored at index controller_id
        for config in controller_configs:
            c = DQNController(config,self.env)
            self.controllers.append(c)
        for controller in self.controllers:
            controller.set_controller_tree(self.controllers)


    def train(self, train_config):
        controllers[0].act(self.env)

        episodes = train_config['episodes']
        for ep in range(episodes):
            controllers[0].act(self.env,0)
        
        for c in controllers
            c.train(controllers)
            

if __name__ == "__main__":
    main()
