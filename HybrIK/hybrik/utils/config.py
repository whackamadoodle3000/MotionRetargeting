import yaml
from easydict import EasyDict as edict


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.safe_load(f))
        return config
