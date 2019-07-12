import pytest
import numpy as np
import newick
from numpy.testing import assert_allclose
from pylo.topology import TreeTopology
from pylo.tree.coalescent import CoalescentTree, ConstantPopulationFunction, GridPopulationFunction

@pytest.fixture
def heterochronous_newick():
    return "(((A:0.2,B:0.1):0.3,C:0.1):0.3,D:0.8)"

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
def test_coalescent_heterochronous(pop,logp_expected,heterochronous_newick):
    tree = newick.loads(heterochronous_newick)[0]
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

skygrid_cases = [
    ([10.0, 15.0, 12.5, 8.0, 7.0], 1.0, -7.009005278982137),
    ([2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0], 2.0, -3.444917777278568),
    ([112.0, 10.2, 12.3], 1.5, -11.774555280807585)
]
@pytest.mark.parametrize('population_sizes,cutoff,logp_expected', skygrid_cases)
def test_coalescent_skygrid(heterochronous_newick, population_sizes, cutoff, logp_expected):
    tree = newick.loads(heterochronous_newick)[0]
    topology = TreeTopology(tree)
    grid_size = len(population_sizes) - 1
    population_function = GridPopulationFunction(topology, population_sizes, cutoff, grid_size)
    height_dist = CoalescentTree.dist(topology, population_function)
    height_values = topology.get_init_heights()[topology.node_mask]
    logp = height_dist.logp(height_values).eval()
    assert_allclose(logp, logp_expected)

