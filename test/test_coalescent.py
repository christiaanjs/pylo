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
    assert_allclose(logp, logp_expected, rtol=1e-3)
    
test_data = [(123,-14.456065261239203),(999,-20.722666738348064)]
@pytest.mark.parametrize('pop,logp_expected', test_data)
def test_coalescent_heterochronous(pop,logp_expected):
    newick_string = "(((A:0.2,B:0.1):0.3,C:0.1):0.3,D:0.8)"
    tree = newick.loads(newick_string)[0]
    topology = TreeTopology(tree)
    population_function = ConstantPopulationFunction(topology, pop)
    height_dist = CoalescentTree.dist(topology, population_function)
    height_values = topology.get_init_heights()[topology.node_mask]
    logp = height_dist.logp(height_values).eval()
    assert_allclose(logp, logp_expected, rtol=1e-3)
