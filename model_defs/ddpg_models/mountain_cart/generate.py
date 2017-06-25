import os
import sys
import getopt
import json
import actor
import critic
from shutil import copyfile

def main(argv):
    """Specify input to generator with:
    -s : save path 
    """
    opts, args = getopt.getopt(argv,"s:")
    save_location = "./models/mountain_cart_DDPG"

    print(opts)
    for opt, arg in opts:
        if opt == "-s":
            save_location = arg

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    cdir = os.path.dirname(__file__)
    conf_path = os.path.join(cdir, 'config.json')

    json_data = open(conf_path).read() 
    config_dict = json.loads(json_data)
    
    print(config_dict)
    adim = config_dict["env"]["action_dim"]
    sdim = config_dict["env"]["state_dim"]
    actor.generate(os.path.join(save_location, config_dict["agent"]["actor_path"]), sdim, adim)
    critic.generate(os.path.join(save_location, config_dict["agent"]["critic_path"]), sdim + adim, 1)
    copyfile(conf_path, os.path.join(save_location, 'config.json'))

if __name__ == "__main__":
    main(sys.argv[1:])
