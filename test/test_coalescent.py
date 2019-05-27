import pytest
import numpy as np
import newick
from numpy.testing import assert_allclose
from pylo.topology import TreeTopology
from pylo.tree.coalescent import CoalescentTree, ConstantPopulationFunction

def test_coalescent_constant_population():
    pop = 10000
    newick_string = "((A:1.0,B:1.0):1.0,C:2.0);"
    tree = newick.loads(newick_string)[0]
    topology = TreeTopology(tree)
    population_function = ConstantPopulationFunction(topology, pop)
    height_dist = CoalescentTree.dist(topology, population_function)
    height_values = topology.get_init_heights()[topology.node_mask]
    logp = height_dist.logp(height_values).eval()
    logp_expected = -(4 / pop) - 2 * np.log(pop)
    assert_allclose(logp, logp_expected)
    
test_data = [(123,-14.446309163678226),(999,-20.721465537146862)]
@pytest.mark.parametrize('pop,logp_expected', test_data)
def test_coalescent_heterochronous(pop,logp_expected):
    newick_string = "(((A:0.2,B:0.1):0.3,C:0.1):0.3,D:0.8)"
    tree = newick.loads(newick_string)[0]
    topology = TreeTopology(tree)
    population_function = ConstantPopulationFunction(topology, pop)
    height_dist = CoalescentTree.dist(topology, population_function)
    height_values = topology.get_init_heights()[topology.node_mask]
    logp = height_dist.logp(height_values).eval()
    assert_allclose(logp, logp_expected)

dengue_cases = [(10, -48.521926838680535),(100, -74.85077951088705), (1000, -110.64089011722196)] 
@pytest.mark.parametrize('pop_size,logp_expected', dengue_cases)
def test_coalescent_dengue(dengue_config, pop_size, logp_expected):
    tree = newick.loads(dengue_config['newick_string'])[0]
    topology = TreeTopology(tree)
    population_function = ConstantPopulationFunction(topology, pop_size)
    height_dist = CoalescentTree.dist(topology, population_function)
    height_values = topology.get_init_heights()[topology.node_mask]
    logp = height_dist.logp(height_values).eval()
    assert_allclose(logp, logp_expected)
