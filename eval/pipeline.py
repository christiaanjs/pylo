import os
import yaml
import sys
import templating
import variational_analysis
import random
import subprocess
import process_results
import numpy as np
import pandas as pd
import pymc3 as pm

def cmd_kwargs(**kwargs):
    pairs = [("-"+ key, str(value)) for key, value in kwargs.items()]
    return [x for pair in pairs for x in pair]

def run_pipeline(**config):
    random.seed(config['seed'])
    pm.set_tt_rng(config['seed'])
    np.random.seed(config['seed'])

    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    build_templates = templating.TemplateBuilder(out_dir)

    beast_args = ['java'] + cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']
    
    pop_size, taxon_names, date_trait_string = build_templates.build_tree_sim(config)
    
    print('Running tree simulation')
    subprocess.run(beast_args + [build_templates.tree_sim_out_path])

    newick_string = build_templates.extract_newick_string()
    build_templates.build_seq_sim(config, taxon_names, newick_string)

    print('Running sequence simulation')
    subprocess.run(beast_args + [build_templates.seq_sim_out_path])

    sequence_dict = build_templates.extract_sequence_dict()
    
    print('Running variational analysis')
    model, inference, pymc_result = variational_analysis.run_analysis(config, newick_string, sequence_dict, build_templates.pymc_analysis_result_path)

    print('Running MCMC analysis')
    mcmc_result = variational_analysis.run_mcmc(config, model, build_templates.nuts_trace_path)
    
    build_templates.build_beast_analysis(config, newick_string, date_trait_string, sequence_dict)
    print('Running BEAST analysis')
    #subprocess.run(beast_args + [str(build_templates.beast_analysis_out_path)])

    #beast_scores = process_results.get_beast_scores(build_templates.beast_analysis_trace_path, config, pop_size).assign(method='beast')
    mcmc_scores = process_results.get_mcmc_scores(mcmc_result, config, pop_size).assign(method='mcmc')
    variational_scores = process_results.get_variational_scores(pymc_result, config, model, inference, pop_size).assign(method='advi')
    combined_scores = pd.concat([mcmc_scores, variational_scores])
    combined_scores.to_csv(build_templates.run_results_path)

    run_summary = {
        'config': config,
        'pop_size': pop_size,
        'date_trate_string': date_trait_string,
        'newick_string': newick_string
    }

    with(open(build_templates.run_summary_path, 'w')) as f:
        yaml.dump(run_summary, f)

if __name__ == '__main__':
    config_filename = sys.argv[1]
    with open(config_filename) as f:
        config = yaml.load(f)
    for i in range(config['n_runs']):
        run_pipeline(out_dir='out/' + str(i) , seed=i+1, **config)
