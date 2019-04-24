import pathlib
import random
import jinja2
from Bio import Phylo
import io
import xml

template_dir = 'templates'
out_dir = 'out'
out_path = pathlib.Path(out_dir)

template_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

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
    
    return pop_size, taxon_names, date_trait_string

seq_sim_template_file = 'sim-seq.j2.xml'
seq_sim_out_file = 'sim-seq.xml'
seq_sim_out_path = out_path / seq_sim_out_file
seq_sim_result_file = 'sequences.xml'
seq_sim_result_path = out_path / seq_sim_result_file


def extract_newick_string():
    with io.StringIO() as s:
        Phylo.convert(tree_sim_result_path, 'nexus', s, 'newick')
        newick_string = s.getvalue().strip()
    return newick_string

def build_seq_sim(config, taxon_names, newick_string):
    seq_sim_template = template_env.get_template(seq_sim_template_file)
    seq_sim_string = seq_sim_template.render(
        taxon_names=taxon_names,
        newick_string=newick_string,
        out_file=seq_sim_result_path, **config)

    with open(seq_sim_out_path, 'w') as f:
        f.write(seq_sim_string)

seq_sim_result_file = 'sequences.xml'
seq_sim_result_path = out_path / seq_sim_result_file

def extract_sequence_dict():
    seq_xml_root = xml.etree.ElementTree.parse(seq_sim_result_path)
    sequence_dict = { tag.attrib['taxon']: tag.attrib['value'] for tag in seq_xml_root.findall('./sequence') }
    return sequence_dict

beast_analysis_template_file = 'beast-analysis.j2.xml'
beast_analysis_out_file = 'beast-analysis.xml'
beast_analysis_out_path = out_path / beast_analysis_out_file

beast_analysis_tree_file = 'beast-log.trees'
beast_analysis_tree_path = out_path / beast_analysis_tree_file
beast_analysis_trace_file = 'beast-log.log'
beast_analysis_trace_path = out_path / beast_analysis_trace_file

def build_beast_analysis(config, newick_string, date_trait_string, sequence_dict):
    beast_analysis_template = template_env.get_template(beast_analysis_template_file)
    beast_analysis_string = beast_analysis_template.render(
        newick_string=newick_string,
        sequence_dict=sequence_dict,
        date_trait_string=date_trait_string,
        trace_out_path=beast_analysis_trace_path,
        tree_out_path=beast_analysis_tree_path,
        **config
    )

    with open(beast_analysis_out_path, 'w') as f:
        f.write(beast_analysis_string)

pymc_analysis_result_file = 'pymc_tracker.pickle'
pymc_analysis_result_path = out_path / pymc_analysis_result_file
