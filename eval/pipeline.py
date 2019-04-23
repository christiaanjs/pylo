import yaml
import sys
import build_templates
import random
import subprocess

def cmd_kwargs(**kwargs):
    pairs = [("-"+ key, str(value)) for key, value in kwargs.items()]
    return [x for pair in pairs for x in pair]

if __name__ == '__main__':
    config_filename = sys.argv[1]
    with open(config_filename) as f:
        config = yaml.load(f)
    
    random.seed(config['seed'])
    beast_args = ['java'] + cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']
    
    pop_size, taxon_names = build_templates.build_tree_sim(config)
    
    # Run tree simulation
    subprocess.run(beast_args + [build_templates.tree_sim_out_path])
   
    build_templates.build_seq_sim(config, taxon_names)
    subprocess.run(beast_args + [build_templates.seq_sim_out_path])
