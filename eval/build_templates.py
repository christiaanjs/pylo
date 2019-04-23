import pathlib
import random
import jinja2
from Bio import Phylo
import io

template_dir = 'templates'
out_dir = 'out'
out_path = pathlib.Path(out_dir)

template_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

###################
# Tree Simulation #
###################

tree_sim_template_file = 'sim-tree.j2.xml'
tree_sim_out_file = 'sim-tree.xml'
tree_sim_out_path = out_path / tree_sim_out_file

tree_sim_result_file = 'sim-tree.trees'
tree_sim_result_path = out_path / tree_sim_result_file

def build_tree_sim(config):
    min_pop_size, max_pop_size, sampling_window, n_taxa = config['min_pop_size'], config['max_pop_size'], config['sampling_window'], config['n_taxa'] 
    pop_size = int(random.random() * (max_pop_size - min_pop_size) + min_pop_size)
    sampling_times = [random.random() * sampling_window for i in range(n_taxa)]
    taxon_names = ["T{}".format(i) for i in range(n_taxa)]
    date_trait_string = ','.join(['{0}={1}'.format(taxon_name, sampling_time) for taxon_name, sampling_time in zip(taxon_names, sampling_times)])

    tree_sim_template = template_env.get_template(tree_sim_template_file)
    tree_sim_string = tree_sim_template.render(pop_size=pop_size, date_trait_string=date_trait_string, taxon_names=taxon_names, out_file=tree_sim_result_path)  


    with open(tree_sim_out_path, 'w') as f:
        f.write(tree_sim_string)
    
    return pop_size, taxon_names

#######################
# Sequence Simulation #
#######################

seq_sim_template_file = 'sim-seq.j2.xml'
seq_sim_out_file = 'sim-seq.xml'
seq_sim_out_path = out_path / seq_sim_out_file
seq_sim_result_file = 'sequences.xml'
seq_sim_result_path = out_path / seq_sim_result_file

def build_seq_sim(config, taxon_names):
    with io.StringIO() as s:
        Phylo.convert(tree_sim_result_path, 'nexus', s, 'newick')
        newick_string = s.getvalue().strip()
        
    seq_sim_template = template_env.get_template(seq_sim_template_file)
    seq_sim_string = seq_sim_template.render(
        taxon_names=taxon_names,
        newick_string=newick_string,
        out_file=seq_sim_result_path, **config)

    with open(seq_sim_out_path, 'w') as f:
        f.write(seq_sim_string)
#
###############################
## Extract sequences from XML #
###############################
#
#seq_sim_result_file = 'sequences.xml'
#
#import xml
#seq_xml_root = xml.etree.ElementTree.parse(out_path / seq_sim_result_file)
#sequence_dict = { tag.attrib['taxon']: tag.attrib['value'] for tag in seq_xml_root.findall('./sequence') }
#
###################
## BEAST Analysis #
###################
#
## TODO: log_every
## prior_params[clock_rate, kappa, pop_size][m, s]i (1.0, 1.25)
## init_values
## estimate_topology
#
#chain_length = int(1e6)
#log_every = 1000
#prior_params = {
#    'clock_rate': { 'm': 1.0, 's': 1.25 },
#    'pop_size': { 'm': 1.0, 's': 1.25 },
#    'kappa': { 'm': 1.0, 's': 1.25 }
#}
#init_values = { 'clock_rate': 1.0, 'pop_size': 1.0, 'kappa': 2.0 }
#estimate_topology = False
#
#beast_analysis_template_file = 'beast-analysis.j2.xml'
#beast_analysis_out_file = 'beast-analysis.xml'
#
## TODO: Re-initialise tree heights
#
#beast_analysis_template = template_env.get_template(beast_analysis_template_file)
#beast_analysis_string = beast_analysis_template.render(
#    pop_size=pop_size,
#    newick_string=newick_string,
#    sequence_dict=sequence_dict,
#    date_trait_string=date_trait_string,
#    chain_length=chain_length,
#    log_every=log_every,
#    init_values=init_values,
#    prior_params=prior_params
#)
#
#with open(out_path / beast_analysis_out_file, 'w') as f:
#    f.write(beast_analysis_string)
#
#
#######################
## PyMC Analysis file #
#######################
#
#import json
#
#pymc_analysis_out_file = 'pymc_analysis.json'
#pymc_analysis_result_file = str(out_path / 'pymc_results.pickle')
#
#inference = 'mean_field' # or full rank, normalising flow
#n_iter = 5000
#n_eval_samples = 1000
#
#config_dict = {
#    'pop_size': pop_size,
#    'newick_string': newick_string,
#    'sequence_dict': sequence_dict,
#    'prior_params': prior_params,
#    'inference': inference,
#    'n_iter': n_iter,
#    'out_file': pymc_analysis_result_file,
#    'n_eval_samples': n_eval_samples
#}
#
#with open(out_path / pymc_analysis_out_file, 'w') as f:
#    json.dump(config_dict, f)
