import yaml
import sys
import build_templates
import variational_analysis
import random
import subprocess

def cmd_kwargs(**kwargs):
    pairs = [("-"+ key, str(value)) for key, value in kwargs.items()]
    return [x for pair in pairs for x in pair]

def run_pipeline(config):
    random.seed(config['seed'])
    beast_args = ['java'] + cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']
    
    pop_size, taxon_names, date_trait_string = build_templates.build_tree_sim(config)
    
    print('Running tree simulation')
    #subprocess.run(beast_args + [build_templates.tree_sim_out_path])

    newick_string = build_templates.extract_newick_string()
    build_templates.build_seq_sim(config, taxon_names, newick_string)

    print('Running sequence simulation')
    #subprocess.run(beast_args + [build_templates.seq_sim_out_path])

    sequence_dict = build_templates.extract_sequence_dict()
    #build_templates.build_beast_analysis(config, newick_string, date_trait_string, sequence_dict)
    print('Running BEAST analysis')
    #subprocess.run(beast_args + [build_templates.beast_analysis_out_path])

    print('Running variational analysis')
    variational_analysis.run_analysis(config, newick_string, sequence_dict, build_templates.pymc_analysis_result_path)

if __name__ == '__main__':
    config_filename = sys.argv[1]
    with open(config_filename) as f:
        config = yaml.load(f)
    run_pipeline(config)
