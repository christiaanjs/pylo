import sys
import json

config_filename = sys.argv[1]

with open(config_filename) as f:
    config = json.load(f)     

from pylo.topology import TreeTopology
import pylo.transform
import newick

tree = newick.loads(config['newick_string'])[0]
topology = TreeTopology(tree)

sequence_dict = config['sequence_dict']
sequence_dict_encoded = pylo.transform.encode_sequences(sequence_dict)
pattern_dict, pattern_counts = pylo.transform.group_sequences(sequence_dict_encoded)
child_patterns = topology.build_sequence_table(pattern_dict)

import pymc3 as pm
from pylo.tree.coalescent import CoalescentTree, ConstantPopulationFunction
from pylo.hky import HKYSubstitutionModel
from pylo.pruning import LeafSequences

def get_lognormal_params(var):
    return { 'mu': config['prior_params'][var]['m'], 'sd': config['prior_params'][var]['s'] }

with pm.Model() as model:
    pop_size = pm.LogNormal('pop_size', **get_lognormal_params('pop_size'))
    pop_func = ConstantPopulationFunction(topology, pop_size)
    tree_heights = CoalescentTree('tree', topology, population_function)
    
    kappa = pm.Lognormal('kappa', **get_lognormal_params('kappa'))
    pi = pm.Dirichlet('pi', a=np.ones(4))
    substitution_model = HKYSubstitutionModel(kappa, pi)

    branch_lengths = topology.get_child_branch_lengths(tree_heights)
    sequences = LeafSequences('sequences', topology, substitution_model, branch_lengths, child_patterns, pattern_counts)


inference = {
    'mean_field': pm.ADVI,
    'full_rank': pm.FullRank
}()


