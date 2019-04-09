import pathlib
import random
import jinja2

template_dir = 'templates'
out_dir = 'out'
out_path = pathlib.Path(out_dir)

template_env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))

###################
# Tree Simulation #
###################

tree_sim_template_file = 'sim-tree.j2.xml'
tree_sim_out_file = 'sim-tree.xml'

n_taxa = 10
min_pop_size = 1000
max_pop_size = 2000
sampling_window = 100

pop_size = int(random.random() * (max_pop_size - min_pop_size) + min_pop_size)
sampling_times = [random.random() * sampling_window for i in range(n_taxa)]
taxon_names = ["T{}".format(i) for i in range(n_taxa)]
date_trait_string = ','.join(['{0}={1}'.format(taxon_name, sampling_time) for taxon_name, sampling_time in zip(taxon_names, sampling_times)])

tree_sim_template = template_env.get_template(tree_sim_template_file)
tree_sim_string = tree_sim_template.render(pop_size=pop_size, date_trait_string=date_trait_string, taxon_names=taxon_names)  

tree_sim_out_path = out_path / tree_sim_out_file
with open(tree_sim_out_path, 'w') as f:
    f.write(tree_sim_string)

#######################
# Sequence Simulation #
#######################

from Bio import Phylo
import io

seq_sim_template_file = 'sim-seq.j2.xml'
tree_sim_result_path = tree_sim_out_path.with_suffix('.trees')
seq_sim_out_file = 'sim-seq.xml'

sequence_length = 20
relaxed_clock = False
mutation_rate = 3.0
kappa = 2.0
frequencies = [0.25, 0.25, 0.25, 0.25]
rate_sd = 0.5

with io.StringIO() as s:
    Phylo.convert(tree_sim_result_path, 'nexus', s, 'newick')
    newick_string = s.getvalue().strip()
    

seq_sim_template = template_env.get_template(seq_sim_template_file)
seq_sim_string = seq_sim_template.render(
    taxon_names=taxon_names,
    newick_string=newick_string,
    relaxed_clock=relaxed_clock,
    sequence_length=sequence_length,
    mutation_rate=mutation_rate,
    kappa=kappa,
    frequencies=frequencies,
    rate_sd=rate_sd)

with open(out_path / seq_sim_out_file, 'w') as f:
    f.write(seq_sim_string)

##############################
# Extract sequences from XML #
##############################

seq_sim_result_file = 'sequences.xml'

import xml
seq_xml_root = xml.etree.ElementTree.parse(out_path / seq_sim_result_file)
sequence_dict = { tag.attrib['taxon']: tag.attrib['value'] for tag in seq_xml_root.findall('./sequence') }

##################
# BEAST Analysis #
##################

# TODO: log_every
# prior_params[clock_rate, kappa, pop_size][m, s]i (1.0, 1.25)
# init_values
# estimate_topology

chain_length = int(1e6)
log_every = 1000
prior_params = {
    'clock_rate': { 'm': 1.0, 's': 1.25 },
    'pop_size': { 'm': 1.0, 's': 1.25 },
    'kappa': { 'm': 1.0, 's': 1.25 }
}
init_values = { 'clock_rate': 1.0, 'pop_size': 1.0, 'kappa': 2.0 }
estimate_topology = False

beast_analysis_template_file = 'beast-analysis.j2.xml'
beast_analysis_out_file = 'beast-analysis.xml'

# TODO: Re-initialise tree heights

beast_analysis_template = template_env.get_template(beast_analysis_template_file)
beast_analysis_string = beast_analysis_template.render(
    newick_string=newick_string,
    sequence_dict=sequence_dict,
    date_trait_string=date_trait_string,
    chain_length=chain_length,
    log_every=log_every,
    init_values=init_values,
    prior_params=prior_params
)

with open(out_path / beast_analysis_out_file, 'w') as f:
    f.write(beast_analysis_string)
