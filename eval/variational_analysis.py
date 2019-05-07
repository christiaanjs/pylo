import numpy as np
import theano.tensor as tt
from pylo.topology import TreeTopology
import pylo.transform
import newick
import pymc3 as pm
from pylo.tree.coalescent import CoalescentTree, ConstantPopulationFunction
from pylo.hky import HKYSubstitutionModel
from pylo.pruning import LeafSequences
import sys
import json
import datetime
import pickle

def construct_model(config, tree, sequence_dict):
    topology = TreeTopology(tree)
    sequence_dict_encoded = pylo.transform.encode_sequences(sequence_dict)
    pattern_dict, pattern_counts = pylo.transform.group_sequences(sequence_dict_encoded)
    pattern_counts = tt.as_tensor_variable(pattern_counts)
    child_patterns = tt.as_tensor_variable(topology.build_sequence_table(pattern_dict))

    def get_lognormal_params(var):
        return { 'mu': config['prior_params'][var]['m'], 'sd': config['prior_params'][var]['s'] }

    with pm.Model() as model:
        pop_size = pm.Lognormal('pop_size', **get_lognormal_params('pop_size'))
        pop_func = ConstantPopulationFunction(topology, pop_size)
        tree_heights = CoalescentTree('tree', topology, pop_func)
        
        kappa = pm.Lognormal('kappa', **get_lognormal_params('kappa'))
        pi = pm.Dirichlet('pi', a=np.ones(4))
        substitution_model = HKYSubstitutionModel(kappa, pi)
        clock_rate = pm.Lognormal('clock_rate', **get_lognormal_params('clock_rate'))

        branch_lengths = topology.get_child_branch_lengths(tree_heights)
        distances = branch_lengths * clock_rate
        sequences = LeafSequences('sequences', topology, substitution_model, distances, child_patterns, pattern_counts)
    return model

class SampleTracker(pm.callbacks.Tracker):
    def __init__(self, save_every=1, *args, **kwargs):
        self.save_every = save_every
        super().__init__(*args, **kwargs)

    def record(self, approx, hist, i):
        if i % self.save_every == 0:
            super().record(approx, hist, i)

    __call__ = record

def construct_inference(config, model):
    return {
        'mean_field': pm.ADVI,
        'full_rank': pm.FullRankADVI
    }[config['inference']](model=model)

def run_analysis(config, newick_string, sequence_dict, out_file):
    tree = newick.loads(newick_string)[0]
    model = construct_model(config, tree, sequence_dict)
    inference = construct_inference(config, model)

    tracker = SampleTracker(
        save_every=config['log_every'],
        i=lambda approx, hist, i: i,
        date_time=datetime.datetime.now,
        **{ key: value.eval for key, value in inference.approx.shared_params.items() }
    )

    inference.fit(n=config['n_iter'], callbacks=[tracker])
    
    with open(out_file, 'wb') as f:     
        pickle.dump(tracker.hist, f)

    return model, inference, tracker.hist

class TimedTrace(pm.backends.NDArray):
    def setup(self, draws, chain, sampler_vars=None):
        super().setup(draws, chain, sampler_vars=sampler_vars)
        self.times = np.empty(draws, dtype='datetime64[s]')
        
    def record(self, point, sampler_stats=None):
        self.times[self.draw_idx] = np.datetime64(datetime.datetime.now())
        super().record(point, sampler_stats=sampler_stats)
        
    def close(self):
        super().close()
        if self.draw_idx == self.draws:
            return
        self.times = self.times[:self.draw_idx]
        

def run_mcmc(config, newick_string, sequence_dict, out_file):
    tree = newick.loads(newick_string)[0]
    model = construct_model(config, tree, sequence_dict)
    with model:
        trace = TimedTrace()
        step = pm.Metropolis()
        pm.sample(chains=1, draws=config['chain_length'], trace=trace, step=[step], tune=0)
   
    with open(out_file, 'wb') as f:
        pickle.dump(trace, f) 

    return trace
