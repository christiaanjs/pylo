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

test_data = [tree_vals + (use_max,) for use_max in [True, False] for tree_vals in
    [ # Tree, proportions, no max root val, optional max height, max height root val
        ('((:0.4,:0.2):0.6,(:0.3,:0.5):0.5)', [0.25, 0.375], 0.8, 2.2, 0.4),
        ('(((:0.2,:0.1):0.3,:0.1):0.3,:0.8)', [0.25, 0.25], 0.4, 1.2, 0.5)
    ]
]

def unconstrain_proportions(constrained_proportions, root_val, use_max, max_root_val):
    return np.concatenate((logit(constrained_proportions), [(logit(max_root_val) if use_max else np.log(root_val))]))


@pytest.mark.parametrize('newick_string,constrained_proportions,root_val,use_max,max_height,max_root_val', test_data)
def test_tree_height_proportion_transform_forward(newick_string, constrained_proportions, root_val, use_max, max_height, max_root_val):
    newick_parsed = newick.loads(newick_string)[0]
    topology = TreeTopology(newick_parsed)
    node_heights = topology.get_init_heights()
    transform = TreeHeightProportionTransform(topology, max_height=(max_height if use_max else None))
    node_heights_ = tt.vector()
    transformed_ = transform.forward(node_heights_)
    transform_func = theano.function([node_heights_], transformed_)
    transformed = transform_func(node_heights)

    expected = unconstrain_proportions(constrained_proportions, root_val, use_max, max_root_val)
    assert_allclose(transformed, expected)


@pytest.mark.parametrize('newick_string,constrained_proportions,root_val,use_max,max_height,max_root_val', test_data)
def test_tree_height_proportion_transform_backward(newick_string, constrained_proportions, root_val, use_max, max_height, max_root_val):
    newick_parsed = newick.loads(newick_string)[0]
    topology = TreeTopology(newick_parsed)
    expected_node_heights = topology.get_init_heights()
    transform = TreeHeightProportionTransform(topology, max_height=(max_height if use_max else None))
    transformed_ = tt.vector()
    node_heights_ = transform.backward(transformed_)
    transform_func = theano.function([transformed_], node_heights_)
    node_heights = transform_func(unconstrain_proportions(constrained_proportions, root_val, use_max, max_root_val))

    assert_allclose(node_heights, expected_node_heights)


@pytest.mark.parametrize('newick_string,constrained_proportions,root_val,use_max,max_height,max_root_val', test_data)
def test_tree_height_proportion_transform_jacobian(newick_string, constrained_proportions, root_val, use_max, max_height, max_root_val):
    newick_parsed = newick.loads(newick_string)[0]
    topology = TreeTopology(newick_parsed)
    transform = TreeHeightProportionTransform(topology, max_height=(max_height if use_max else None))
    transformed_ = tt.vector()
    node_heights_ = transform.backward(transformed_)
    jac_ = theano.gradient.jacobian(node_heights_, transformed_)
    jac_func = theano.function([transformed_], jac_)
    jac_expected = jac_func(unconstrain_proportions(constrained_proportions, root_val, use_max, max_root_val))
    log_det_jac_ = transform.jacobian_det(transformed_)
    log_det_jac_func = theano.function([transformed_], log_det_jac_)
    log_det_jac = log_det_jac_func(unconstrain_proportions(constrained_proportions, root_val, use_max, max_root_val))
    assert_allclose(log_det_jac, log_det_jac_expected)
