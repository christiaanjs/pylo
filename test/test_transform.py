import theano
import theano.tensor as tt
import pytest
import numpy as np
from numpy.testing import assert_allclose
import newick
from scipy.special import logit

from pylo.topology import TreeTopology
from pylo.tree import TreeHeightProportionTransform

test_data = [
    ('((:0.3,:0.3):0.7,(:0.5,:0.5):0.5)', [0.3, 0.5, 1.0]),
    ('(((:0.2,:0.2):0.6,:0.8):0.2,:1.0)', [0.25, 0.8, 1.0])
]
@pytest.mark.parametrize('newick_string,constrained_proportions', test_data)
def test_tree_height_proportion_transform_forward(newick_string, constrained_proportions):
    newick_parsed = newick.loads(newick_string)[0]
    topology = TreeTopology(newick_parsed)
    node_heights = topology.get_init_heights()
    transform = TreeHeightProportionTransform(topology)
    node_heights_ = tt.vector()
    transformed_ = transform.forward(node_heights_)
    transform_func = theano.function([node_heights_], transformed_)
    transformed = transform_func(node_heights)

    expected = np.concatenate((logit(constrained_proportions[:-1]), np.log(constrained_proportions[-1:])))
    assert_allclose(transformed, expected)
