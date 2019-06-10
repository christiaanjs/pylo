import random
import pymc3 as pm
import numpy as np

def update_dict(dict, **kwargs):
    res = dict.copy()
    res.update(**kwargs)
    return res

def cmd_kwargs(**kwargs):
    pairs = [("-"+ key, str(value)) for key, value in kwargs.items()]
    return [x for pair in pairs for x in pair]

def get_beast_args(config):
    return ['java'] + cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']

def do_seeding(config):
    random.seed(config['seed'])
    pm.set_tt_rng(config['seed'])
    np.random.seed(config['seed'])
