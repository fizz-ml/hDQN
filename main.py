from environment import GymEnvironment
from ddpg_agent import DDPGAgent
import numpy as np
import sys
import random
import os
import json
from dqn_controller import DQNController

def main():
    json_data = open(sys.argv[1]).read()
    conf = json.loads(json_data)


    run = Runner(conf['env'], conf['subroutines'])
    run.train(conf['train'])

class Runner:
    def __init__(self, env_config, subroutine_configs):
        self.env = GymEnvironment(name = env_config["name"])
        self.controllers = []
        for config in subroutine_configs:
            c = DQNController(config,self.env)
            self.controllers.append(c)

    def train(self, train_config):
        controllers[0].act(self.env)

        episodes = train_config['episodes']
        for ep in range(episodes):
            controllers[0].act(self.env,0)
        
        for c in controllers
            c.train(controllers)
            

if __name__ == "__main__":
    main()
