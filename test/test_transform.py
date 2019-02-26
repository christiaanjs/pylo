import theano
import theano.tensor as tt
import pytest
import numpy as np
from numpy.testing import assert_allclose
import newick
from scipy.special import logit
from scipy.linalg import det

from pylo.topology import TreeTopology
from pylo.tree import TreeHeightProportionTransform

test_data = [
    ('((:0.3,:0.3):0.7,(:0.5,:0.5):0.5)', [0.3, 0.5, 1.0]),
    ('(((:0.2,:0.2):0.6,:0.8):0.2,:1.0)', [0.25, 0.8, 1.0])
]

def unconstrain_proportions(constrained_proportions):
    return np.concatenate((logit(constrained_proportions[:-1]), np.log(constrained_proportions[-1:])))


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

    expected = unconstrain_proportions(constrained_proportions)
    assert_allclose(transformed, expected)


@pytest.mark.parametrize('newick_string,constrained_proportions', test_data)
def test_tree_height_proportion_transform_backward(newick_string, constrained_proportions):
    newick_parsed = newick.loads(newick_string)[0]
    topology = TreeTopology(newick_parsed)
    expected_node_heights = topology.get_init_heights()
    transform = TreeHeightProportionTransform(topology)
    transformed_ = tt.vector()
    node_heights_ = transform.backward(transformed_)
    transform_func = theano.function([transformed_], node_heights_)
    node_heights = transform_func(unconstrain_proportions(constrained_proportions))

    assert_allclose(node_heights, expected_node_heights)


@pytest.mark.parametrize('newick_string,constrained_proportions', test_data)
def test_tree_height_proportion_transform_jacobian(newick_string, constrained_proportions):
    newick_parsed = newick.loads(newick_string)[0]
    topology = TreeTopology(newick_parsed)
    transform = TreeHeightProportionTransform(topology)
    transformed_ = tt.vector()
    node_heights_ = transform.backward(transformed_)
    jac_ = theano.gradient.jacobian(node_heights_, transformed_)
    jac_func = theano.function([transformed_], jac_)
    jac_expected = jac_func(unconstrain_proportions(constrained_proportions))
    log_det_jac_expected = np.log(np.abs(det(jac_expected)))
    log_det_jac_ = transform.jacobian_det(transformed_)
    log_det_jac_func = theano.function([transformed_], log_det_jac_)
    log_det_jac = log_det_jac_func(unconstrain_proportions(constrained_proportions))
    assert_allclose(log_det_jac, log_det_jac_expected)
