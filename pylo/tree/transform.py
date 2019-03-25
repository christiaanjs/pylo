import numpy as np
import theano
import theano.tensor as tt
from pymc3.distributions.transforms import Transform
from pymc3.math import logit, invlogit
from scipy.special import logit as logit_val

class TreeHeightProportionTransform(Transform):
    name = 'tree_height_proportion'

    def __init__(self, topology, max_height=None):
        self.topology = topology
        self.max_height = max_height
        
    def forward(self, x_): # Theano
        non_root_proportions, root_height = self.topology.get_proportions(x_)
        if self.max_height is None:
            root_val = tt.log(root_height)
        else:
            root_val = logit(root_height / (self.max_height - self.topology.get_max_leaf_height()))
        return tt.concatenate([logit(non_root_proportions), tt.stack(root_val)])
    
    def forward_val(self, x_, point=None): # Numpy
        non_root_proportions, root_height = self.topology.get_proportions(x_)
        if self.max_height is None:
            root_val = np.log(root_height)
        else:
            root_val = logit_val(root_height / (self.max_height - self.topology.get_max_leaf_height()))
        return np.concatenate([logit_val(non_root_proportions), [root_val]])
    
    def backward(self, y_):
        if self.max_height is None:
            root_val = tt.exp(y_[-1])
        else:
            root_proportion = invlogit(y_[-1])
            root_val = root_proportion * (self.max_height - self.topology.get_max_leaf_height())
        proportions = invlogit(y_[:-1])
        return self.topology.get_heights(root_val, proportions)

    def jacobian_det(self, y_):
        proportions = invlogit(y_[:-1])
        y_root = y_[-1]
        times = self.backward(y_)

        if self.max_height is None:
            root_contrib = y_root
        else:
            root_proportion = invlogit(y_root)
            root_contrib = tt.log(root_proportion) + tt.log(1 - root_proportion) + tt.log(self.max_height - self.topology.get_max_leaf_height())
        
        constrain_log_jac_det = tt.sum(tt.log(proportions) + tt.log(1 - proportions))
        tree_transform_log_jac_det = tt.sum(tt.log(times[self.topology.node_parent_indices[:-1]] - self.topology.get_max_node_heights()[:-1]))
        return constrain_log_jac_det + tree_transform_log_jac_det + root_contrib
        
        