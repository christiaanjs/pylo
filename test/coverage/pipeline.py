import os
import yaml
import pickle
import sys
sys.path += ['../../eval']
import templating
import variational_analysis
import random
import subprocess
import process_results
import numpy as np
import pandas as pd
import pymc3 as pm
from tqdm import tqdm

def cmd_kwargs(**kwargs):
    pairs = [("-"+ key, str(value)) for key, value in kwargs.items()]
    return [x for pair in pairs for x in pair]

def do_seeding(config):
    random.seed(config['seed'])
    pm.set_tt_rng(config['seed'])
    np.random.seed(config['seed'])
    

def sim_pipeline(**config):
    do_seeding(config)

    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    build_templates = templating.TemplateBuilder(out_dir, template_dir='../../eval/templates')

    beast_args = ['java'] + cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']
    
    pop_size, taxon_names, date_trait_string = build_templates.build_tree_sim(config)
    
    print('Running tree simulation')
    subprocess.run(beast_args + [build_templates.tree_sim_out_path])

    newick_string = build_templates.extract_newick_string()
    build_templates.build_seq_sim(config, taxon_names, newick_string)

    print('Running sequence simulation')
    subprocess.run(beast_args + [build_templates.seq_sim_out_path])

    sequence_dict = build_templates.extract_sequence_dict()
    
    build_templates.build_beast_analysis(config, newick_string, date_trait_string, sequence_dict)
    print('Running BEAST analysis')
    subprocess.run(beast_args + [str(build_templates.beast_analysis_out_path)])

    run_summary = {
        'config': config,
        'pop_size': pop_size,
        'date_trate_string': date_trait_string,
        'newick_string': newick_string
    }

    with(open(build_templates.run_summary_path, 'w')) as f:
        yaml.dump(run_summary, f)

def beast_fixed_pipeline(**config):
    out_dir = config['out_dir']
    build_templates = templating.TemplateBuilder(out_dir, template_dir='../../eval/templates')

    sequence_dict = build_templates.extract_sequence_dict()
    
    with(open(build_templates.run_summary_path)) as f:
        run_summary = yaml.load(f)

    newick_string, date_trait_string = run_summary['newick_string'], run_summary['date_trate_string']

    config['estimate_topology'] = False
    analysis_file = 'fixed-beast-analysis.xml'
    trace_file = 'fixed-beast-log.log'
    tree_file = 'fixed-beast-log.trees'
    analysis_path = build_templates.build_beast_analysis(config, newick_string, date_trait_string, sequence_dict, out_file=analysis_file, trace_file=trace_file, tree_file=tree_file)
    beast_args = ['java'] + cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']
    subprocess.run(beast_args + [str(analysis_path)])


def mcmc_pipeline(**config):
    do_seeding(config)
    build_templates = templating.TemplateBuilder(config['out_dir'])

    sequence_dict = build_templates.extract_sequence_dict()
    
    with(open(build_templates.run_summary_path)) as f:
        run_summary = yaml.load(f)

    variational_analysis.run_mcmc(config, run_summary['newick_string'], sequence_dict, build_templates.nuts_trace_path)

def trace_pipeline(**config):
    do_seeding(config)
    build_templates = templating.TemplateBuilder(config['out_dir'])

    sequence_dict = build_templates.extract_sequence_dict()
    
    with(open(build_templates.run_summary_path)) as f:
        run_summary = yaml.load(f)
    
    #print('Getting variational trace')
    #variational_trace = process_results.get_variational_trace(build_templates.pymc_analysis_result_path, config, sequence_dict, run_summary['newick_string'])
    #variational_trace_df = process_results.process_pymc_trace(variational_trace, config).assign(method='variational')
    print('Getting BEAST trace')
    with open(build_templates.nuts_trace_path, 'rb') as f:
        mcmc_trace = pickle.load(f)
    mcmc_trace_df = process_results.process_pymc_trace(mcmc_trace, config, burn_in=True).assign(method='pymc-mcmc')
    beast_trace_df = process_results.process_beast_trace(build_templates.beast_analysis_trace_path, config).assign(method='beast')
    beast_fixed_trace_df = process_results.process_beast_trace(build_templates.out_path / 'fixed-beast-log.log', config).assign(method='beast-fixed')
    combined_trace = pd.concat([mcmc_trace_df, beast_trace_df, beast_fixed_trace_df])
    combined_trace.to_csv(build_templates.run_trace_path)

if __name__ == '__main__':
    config_filename = sys.argv[1]
    to_call = eval(sys.argv[2])
    
    with open(config_filename) as f:
        config = yaml.load(f)
    for i in tqdm(range(config['n_runs'])):
        to_call(out_dir='out/' + str(i) , seed=i+1, **config)
