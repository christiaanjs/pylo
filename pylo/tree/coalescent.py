import numpy as np
import theano.tensor as tt
from pylo.tree.transform import TreeHeightProportionTransform
from pymc3.distributions import Continuous
from pymc3.util import get_variable_name

def coalescent_likelihood(n_lineages,
                          population_func, # At coalescence
                          population_areas, # Integrals
                          coalescent_mask):
    k_choose_2 = n_lineages * (n_lineages - 1) * 0.5
    return -tt.sum(k_choose_2 * population_areas) - tt.sum(tt.log(population_func[coalescent_mask]))
    
class CoalescentTree(Continuous):
    def __init__(self, topology, population_func, *args, **kwargs):
        shape = topology.get_internal_node_count()
        kwargs.setdefault('shape', shape)
        transform = TreeHeightProportionTransform(topology)
        testval_transformed = np.concatenate([0.5*np.ones(shape - 1), [1.0]])
        testval = transform.backward(testval_transformed).eval() 
        super(BirthDeathSamplingTree, self).__init__(
            testval=testval, transform=transform, *args, **kwargs)

        self.topology = topology
        self.population_func = population_func
        
    def logp(self, value):
        pass
        
    