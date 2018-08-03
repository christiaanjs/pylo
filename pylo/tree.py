import theano.tensor as tt
from pymc3.distributions import Continuous
from pymc3.distributions.transform import Transform
from pymc3.util import get_variable_name
from pymc3.math import logit, invlogit

class TreeHeightProportionTransform(Transform):
    name = 'tree_height_proportion'

    def __init__(self, topology):
        self.topology = topology
        
    def forward(self, x_): # Theano
        pass
    
    def forward_val(self, x_, point=None): # Numpy
        pass
    
    def backward(self, y_):
        root_height = tt.exp(y_[-1]) # Root in last position
        proportions = tt.invlogit(y_[:-1])
        return self.topology.get_heights(root_height, proportions)

    def jacobian_det(self, x):
        pass
        

class BirthDeathSamplingTree(Continuous):
    def __init__(self, topology, lam, mu=0.0, rho=1.0, *args, **kwargs):
        shape = topology.get_n_internal_nodes()
        kwargs.setdefault('shape', shape)
        super(Continuous, self).__init__(transform=TreeHeightProportionTransform(topology), *args, **kwargs)

        self.topology = topology
        self.lam = tt.as_tensor_variable(lam)
        self.mu = tt.as_tensor_variable(mu)
        self.rho = tt.as_tensor_variable(rho)

        self.r = lam - mu
        self.a = mu/lam

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
        

    def _repr_latex(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        lam = dist.lam
        return r'${} \sim \text{{BirthDeathTree}}(\lambda={}, \mu={}'.format(name, get_variable_name(mu), get_variable_name(lam))

    # TODO: def random
