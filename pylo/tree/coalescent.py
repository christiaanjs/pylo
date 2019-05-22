import numpy as np
import theano.tensor as tt
from pylo.tree.transform import TreeHeightProportionTransform
from pymc3.distributions import Continuous
from pymc3.util import get_variable_name

def coalescent_likelihood(lineage_count,
                          population_func, # At coalescence
                          population_areas, # Integrals of 1/N
                          coalescent_mask): # At top of interval
    k_choose_2 = lineage_count * (lineage_count - 1) * 0.5
    return -tt.sum(k_choose_2 * population_areas) - tt.sum(tt.log(population_func[coalescent_mask]))

def get_lineage_count(node_mask):
    return tt.cumsum(tt.where(node_mask, -1, 1))

class PopulationFunction:
    def make_intervals(heights_sorted, node_mask): #returns lineage_count, population_func, population_areas, coalescent_mask
        raise NotImplementedError()
        
class ConstantPopulationFunction(PopulationFunction):
    def __init__(self, topology, population_size):
        self.population_size = population_size
        self.topology = topology
        
    def make_intervals(self, heights_sorted, node_mask):
        lineage_count = get_lineage_count(node_mask)[:-1]
        population_func_array = tt.alloc(self.population_size, self.topology.get_node_count() - 1)
        durations = heights_sorted[1:] - heights_sorted[:-1]
        population_areas = durations / self.population_size
        coalescent_mask = node_mask[1:]
        return lineage_count, population_func_array, population_areas, coalescent_mask
        
    
class CoalescentTree(Continuous):
    def __init__(self, topology, population_func, *args, **kwargs):
        shape = topology.get_internal_node_count()
        kwargs.setdefault('shape', shape)
        transform = TreeHeightProportionTransform(topology)
        testval_transformed = np.concatenate([0.5*np.ones(shape - 1), [1.0]])
        testval = transform.backward(testval_transformed).eval() 
        super(CoalescentTree, self).__init__(
            testval=testval, transform=transform, *args, **kwargs)

        self.topology = topology
        self.population_func = population_func
        
    def logp(self, value):
        heights_sorted, node_mask = self.topology.get_heights_sorted(value)
        lineage_count, population_func, population_areas, coalescent_mask = self.population_func.make_intervals(heights_sorted, node_mask)
        return coalescent_likelihood(lineage_count, population_func, population_areas, coalescent_mask)
        
    
