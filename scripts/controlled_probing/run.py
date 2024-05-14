import random, time, os, pytz, argparse, yaml, sys
sys.path.append("../")

from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs.yaml")
args = parser.parse_args()

with open(args.config_file, 'rb') as f: 
    config = yaml.full_load(f.read()) # dict

config['date'] = date
os.mkdir(os.path.join(config["save_dir"], date))

cur_config_file = os.path.join(config["save_dir"], date, "configs.yaml")
yaml.dump(config, open(cur_config_file, "w"))

os.system("python train.py --config {} | tee {}".format(
    cur_config_file,
    os.path.join(config["save_dir"], date, "terminal.txt"),
))