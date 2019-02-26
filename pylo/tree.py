import numpy as np
import theano
import theano.tensor as tt
from pymc3.distributions import Continuous
from pymc3.distributions.transforms import Transform
from pymc3.util import get_variable_name
from pymc3.math import logit, invlogit
from scipy.special import logit as logit_val

class TreeHeightProportionTransform(Transform):
    name = 'tree_height_proportion'

    def __init__(self, topology):
        self.topology = topology
        
    def forward(self, x_): # Theano
        non_root_proportions, root_height = self.topology.get_proportions(x_)
        return tt.concatenate([logit(non_root_proportions), tt.stack(tt.log(root_height))])
    
    def forward_val(self, x_, point=None): # Numpy
        non_root_proportions, root_height = self.topology.get_proportions(x_)
        return np.concatenate([logit_val(non_root_proportions), [np.log(root_height)]])
    
    def backward(self, y_):
        root_height = tt.exp(y_[-1]) # Root in last position
        proportions = invlogit(y_[:-1])
        return self.topology.get_heights(root_height, proportions)

    def jacobian_det(self, y_):
        proportions = invlogit(y_[:-1])
        y_root = y_[-1]
        times = self.backward(y_)
        constrain_log_jac_det = y_root + tt.sum(tt.log(proportions) + tt.log(1 - proportions))
        tree_transform_log_jac_det = tt.sum(tt.log(times[self.topology.parent_indices[:-1]]))
        return constrain_log_jac_det + tree_transform_log_jac_det
        
        
class BirthDeathSamplingTree(Continuous):
    def __init__(self, topology, r=1.0, a=0.0, rho=1.0, *args, **kwargs):
        shape = topology.get_internal_node_count()
        kwargs.setdefault('shape', shape)
        transform = TreeHeightProportionTransform(topology)
        testval_transformed = np.concatenate([0.5*np.ones(shape - 1), [1.0]])
        testval = transform.backward(testval_transformed).eval() 
        super(BirthDeathSamplingTree, self).__init__(
            testval=testval, transform=transform, *args, **kwargs)

        self.topology = topology
        self.r = tt.as_tensor_variable(r)
        self.a = tt.as_tensor_variable(a)
        self.rho = tt.as_tensor_variable(rho)

        self.mu = self.a*self.r/(1.0-self.a)
        self.lam = self.r + self.mu

        # TODO: think about parameter constraints

    def logp(self, value):
        topology = self.topology
        taxon_count = tt.as_tensor_variable(topology.get_taxon_count())
        root_index = topology.get_root_index()
       
        r = self.r
        a = self.a
        rho = self.rho

        log_coeff = (taxon_count - 1)*tt.log(2.0) - tt.gammaln(taxon_count)
        tree_logp = log_coeff + (taxon_count - 1)*tt.log(r*rho) + taxon_count*tt.log(1 - a)
        
        mrhs = -r*value
        zs = tt.log(rho + ((1 - rho) - a)*tt.exp(mrhs))
        ls = -2*zs + mrhs
        root_term = mrhs[root_index] - zs[root_index]
        
        return tree_logp + tt.sum(ls) + root_term
        
        # TODO: bound with pymc3.distributions.dist_math.bound


    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        r = dist.r
        a = dist.a
        rho = dist.rho
        return r'${} \sim \text{{BirthDeathSamplingTree}}(r={}, a={}, \rho={})'.format(name,
            get_variable_name(r), get_variable_name(a), get_variable_name(rho))

    # TODO: def random
