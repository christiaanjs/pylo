import numpy as np
import theano.tensor as tt
from pylo.tree.transform import TreeHeightProportionTransform
from pymc3.distributions import Continuous
from pymc3.util import get_variable_name

COALESCENCE, SAMPLING, OTHER = -1, 1, 0

def coalescent_likelihood(lineage_count,
                          population_func, # At coalescence
                          population_areas, # Integrals of 1/N
                          coalescent_mask): # At top of interval
    k_choose_2 = lineage_count * (lineage_count - 1) * 0.5
    return -tt.sum(k_choose_2 * population_areas) - tt.sum(tt.log(population_func[coalescent_mask]))

def get_lineage_count(event_types):
    return tt.cumsum(event_types)

class PopulationFunction:
    def make_intervals(heights_sorted, node_mask): #returns lineage_count, population_func, population_areas, coalescent_mask
        raise NotImplementedError()
        
class ConstantPopulationFunction(PopulationFunction):
    def __init__(self, topology, population_size):
        self.population_size = population_size
        self.topology = topology
        
    def make_intervals(self, heights_sorted, node_mask):
        lineage_count = get_lineage_count(tt.where(node_mask, COALESCENCE, SAMPLING))[:-1]
        population_func_array = tt.alloc(self.population_size, self.topology.get_node_count() - 1)
        durations = heights_sorted[1:] - heights_sorted[:-1]
        population_areas = durations / self.population_size
        coalescent_mask = node_mask[1:]
        return lineage_count, population_func_array, population_areas, coalescent_mask

class GridPopulationFunction(PopulationFunction): # Piecewise constant changing over grid
    def __init__(self, topology, population_func, max_height, grid_size):
        self.population_func = population_func
        self.topology = topology
        self.max_height = max_height
        self.grid_size = grid_size

    def make_intervals(self, heights_sorted, node_mask):
        grid_times = self.max_height / self.grid_size * (np.arange(self.grid_size) + 1)

        event_types_unsorted = tt.concatenate((tt.alloc(OTHER, self.grid_size + 1), tt.where(node_mask, COALESCENCE, SAMPLING)))
        times_unsorted = tt.concatenate((grid_times, [np.inf], heights_sorted))
        pop_sizes_unsorted = tt.concatenate((self.population_func, tt.fill(node_mask, np.nan)))
        sort_indices = tt.argsort(times_unsorted, kind='stable')
        indices = np.arange(self.topology.get_node_count() + self.grid_size + 1)

        times_sorted = times_unsorted[sort_indices]
        time_mask = times_sorted <= heights_sorted[-1]
        times_to_use = times_sorted[time_mask]


        event_types_sorted = event_types_unsorted[sort_indices]
        event_types_to_use = event_types_sorted[time_mask]
        lineage_count = get_lineage_count(event_types_to_use)[:-1]
        coalescent_mask = tt.eq(event_types_to_use[1:], COALESCENCE)

        pop_sizes_null = pop_sizes_unsorted[sort_indices]
        notnull_indices = (~tt.isnan(pop_sizes_null) & (indices[:, np.newaxis] <= indices[np.newaxis, :])).argmax(axis=1)
        pop_sizes = pop_sizes_null[notnull_indices][time_mask][1:]
        durations = times_to_use[1:] - times_to_use[:-1]
        population_areas = durations / pop_sizes 
        return lineage_count, pop_sizes, population_areas, coalescent_mask 
    
class CoalescentTree(Continuous):
    def __init__(self, topology, population_func, *args, **kwargs):
        shape = topology.get_internal_node_count()
        kwargs.setdefault('shape', shape)
        transform = TreeHeightProportionTransform(topology)
        super(CoalescentTree, self).__init__(transform=transform, *args, **kwargs)

        self.topology = topology
        self.population_func = population_func
        
    def logp(self, value):
        heights_sorted, node_mask = self.topology.get_heights_sorted(value)
        lineage_count, population_func, population_areas, coalescent_mask = self.population_func.make_intervals(heights_sorted, node_mask)
        return coalescent_likelihood(lineage_count, population_func, population_areas, coalescent_mask)
        
    
