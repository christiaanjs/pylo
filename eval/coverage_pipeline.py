import yaml
import json
import pickle
import newick
import pymc3 as pm
import numpy as np
import scipy
import pandas as pd
import random
import sys
import os
import io
import pathlib
import tqdm
import templating
import variational_analysis
import topology_inference
import subprocess
import process_results
import util
import Bio
import Bio.Phylo
import pylo


def do_coverage(config):
    util.do_seeding(config)
    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    build_templates = templating.TemplateBuilder(out_dir)

    beast_args = ['java'] + util.cmd_kwargs(jar=config['beast_jar'], seed=config['seed']) + ['-overwrite']

    print('Running simulations for seed {0}'.format(config['seed']))
    pop_size, taxon_names, date_trait_string = build_templates.build_tree_sim(config)

    def run_beast(xml_path, **kwargs):
        result = subprocess.run(beast_args + [xml_path], **kwargs)
        if result.returncode != 0:
            print(result.stderr)
            print(result.stdout)
            raise RuntimeError('BEAST run failed')
        else:
            print('Ran BEAST ({0}) successfully'.format(xml_path))


        
    run_beast(build_templates.tree_sim_out_path)
    newick_string = build_templates.extract_newick_string(build_templates.tree_sim_result_path)

    run_summary = {
        'config': config,
        'pop_size': pop_size,
        'date_trait_string': date_trait_string,
        'newick_string': newick_string
    }

    with(open(build_templates.run_summary_path, 'w')) as f:
        yaml.dump(run_summary, f)

    build_templates.build_seq_sim(config, taxon_names, newick_string)
    run_beast(build_templates.seq_sim_out_path)
    sequence_dict = build_templates.extract_sequence_dict()

    print('Running topology inference for seed {0}'.format(config['seed']))
    nj_tree = topology_inference.get_neighbor_joining_tree(sequence_dict)
    topology_inference.build_lsd_inputs(config, build_templates, nj_tree, date_trait_string)
    subprocess.run([config['lsd_executable']] + topology_inference.get_lsd_args(build_templates))
    lsd_tree = topology_inference.extract_lsd_tree(build_templates)    
    analysis_newick_io = io.StringIO()
    Bio.Phylo.write([lsd_tree], analysis_newick_io, format='newick')
    analysis_newick = analysis_newick_io.getvalue()

    print('Running BEAST for seed {0}'.format(config['seed']))
    build_templates.build_beast_analysis(config, analysis_newick, date_trait_string, sequence_dict)
    run_beast(build_templates.beast_analysis_out_path)

    print('Running PyMC for seed {0}'.format(config['seed']))
    tree = newick.loads(analysis_newick)[0]
    model = variational_analysis.construct_model(config, tree, sequence_dict)
    inference = variational_analysis.construct_inference(config, model)

    approx = inference.fit(config['n_iter'])

    with open(build_templates.pymc_analysis_result_path, 'wb') as f:
        pickle.dump(approx, f)

    print('Processing results for seed {0}'.format(config['seed']))

    bio_tree = next(Bio.Phylo.parse(io.StringIO(newick_string), 'newick'))
    tree_height = max(bio_tree.depths().values())

    true_values = {
        'tree_height': tree_height,
        'pop_size': pop_size,
        'kappa': config['kappa']
    }

    quantile_ps = np.array([0.025, 0.975])

    beast_trace_df = process_results.process_beast_trace(build_templates.beast_analysis_trace_path, config, burn_in=True)
    beast_trace_melted = beast_trace_df.melt(var_name='variable', value_name='value')
    beast_quantile_df = beast_trace_melted.groupby('variable').quantile(quantile_ps).reset_index().rename(columns={'level_1': 'quantile'})


    rvs_dict = { rv.name: rv for rv in model.deterministics }
    slices = { name: inference.approx.ordering.by_name[rv.transformed.name].slc for name, rv in rvs_dict.items() }
    indices_dict = { 'tree_height': slices['tree'].stop - 1, 'pop_size': slices['pop_size'].start, 'kappa': slices['kappa'].start  }
    varnames = list(indices_dict.keys())
    indices = np.array(list(indices_dict.values()))
    topology = pylo.topology.TreeTopology(tree)
    transforms = {
        'tree_height': lambda x: np.exp(x) + topology.get_max_leaf_height(),
        'kappa': np.exp,
        'pop_size': np.exp
    }
    transformed_quantiles = scipy.stats.norm.ppf(quantile_ps[:, np.newaxis],
                                                 loc=approx.mean.eval()[np.newaxis, indices],
                                                 scale=approx.std.eval()[np.newaxis, indices])
    quantile_dict = { varname: transforms[varname](transformed_quantiles[:, i]) for i, varname in enumerate(varnames)}
    pymc_quantile_df = pd.DataFrame(dict(quantile=quantile_ps, **quantile_dict)).melt(id_vars=['quantile'], var_name='variable', value_name='value')

    all_quantile_df = pd.concat([beast_quantile_df.assign(method='BEAST'), pymc_quantile_df.assign(method='Variational')])
    result_df = all_quantile_df.assign(seed=config['seed'], truth=all_quantile_df.variable.replace(true_values))
    result_df.to_csv(build_templates.run_results_path)
    return result_df


if __name__ == '__main__':
    config_filename = sys.argv[1]

    with open(config_filename) as f:
        config = yaml.load(f)
    
    out_dir = pathlib.Path(config['out_dir'])
    def run_iteration(i):
        run_out_dir = out_dir / str(i)
        run_seed = i + 1
        run_config = util.update_dict(config, out_dir=str(run_out_dir), seed=run_seed) 
        return do_coverage(run_config)
    
    runs = [run_iteration(i) for i in tqdm.tqdm(range(config['n_runs']))]
    result_df = pd.concat(runs)
    result_df.to_csv(out_dir / 'results.csv', index=False)
