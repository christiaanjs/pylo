import numpy as np
import theano.tensor as tt
from pylo.tree.transform import TreeHeightProportionTransform
from pymc3.distributions import Continuous
from pymc3.util import get_variable_name

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
