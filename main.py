from environment import GymEnvironment
from ddpg_agent import DDPGAgent
import numpy as np
import sys
import random
import os
import json
import pdb

def main():
    model_path = sys.argv[1]
    conf_path = os.path.join(model_path, 'config.json')
    json_data = open(conf_path).read()
    conf = json.loads(json_data)

    # TODO later nicer yeah
    conf["agent"]["actor_path"]     = os.path.join(model_path, conf["agent"]["actor_path"] )
    conf["agent"]["critic_path"]    = os.path.join(model_path, conf["agent"]["critic_path"] )

    run = Runner(conf['env'], conf['agent'])
    run.train(conf['train'])
    run.test(conf['test'])

class Runner:
    def __init__(self, env_config, agent_config):
        self.env = GymEnvironment(name = env_config["name"])
        self.agent = DDPGAgent(action_size = self.env.action_size[0],
                                state_size = self.env.obs_size[0],
                                **agent_config)

    def train(self, train_config, fill_replay = True):
        # Fill experience replay
        self.env.new_episode()
        ma_reward = 0
        if fill_replay:
            prefill = train_config['prefill']

            temp_reward = 0
            temp_done = False
            for step in range(prefill):
                cur_obs = self.env.cur_obs
                cur_obs = np.concatenate((cur_obs,np.array([temp_reward])))
                _ = self.agent.get_next_action(cur_obs)
                cur_action = [random.random()*2.0-1.0]*self.env.action_size[0]
                next_state, reward, done = self.env.next_obs(cur_action, render = True)

                temp_reward = reward
                temp_done = done
                self.agent.log_reward(temp_reward, temp_done)
                ma_reward = ma_reward*0.99 + reward*0.01

        # Start training
        train_steps = train_config['steps']

        temp_reward = 0
        temp_done = True
        for step in range(train_steps):
            cur_obs = self.env.cur_obs
            # TODO: This step probably belongs somewhere else
            cur_obs = np.concatenate((cur_obs,np.array([temp_reward])))
            cur_action = np.squeeze(self.agent.get_next_action(cur_obs), axis=0)
            if (any(np.isnan(cur_obs))):
                pdb.set_trace()
            next_state, reward, done = self.env.next_obs(cur_action, render = True)

            temp_reward = reward
            temp_done = done

            self.agent.log_reward(temp_reward, temp_done)

            self.agent.train()
            ma_reward = ma_reward*0.99 + reward*0.01
            if(step % 1000):
                print(cur_obs, ' ', cur_action, 'Reward:', ma_reward)
                print('Eps',self.agent.epsilon)


    def test(self, test_config):
        test_steps = test_config['steps']

        temp_reward = 0
        temp_done = False
        for step in range(start_train):
            cur_obs = self.env.cur_obs
            cur_obs = np.concatenate((cur_obs,np.array([temp_reward])))
            cur_action = self.agent.get_next_action(cur_obs)
            next_state, reward, done = self.env.next_obs(cur_action, render = True)

if __name__ == "__main__":
    main()
