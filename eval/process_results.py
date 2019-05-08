import numpy as np
import pandas as pd
import theano
import theano.tensor as tt
import tqdm
import newick
import pickle
import variational_analysis
import pymc3 as pm

def get_variational_scores(result, config, model, inference, true_pop_size):
    approx_params = list(inference.approx.shared_params.values())
    distance = abs(model.pop_size - true_pop_size)
    input_vars = tt.dvectors(len(approx_params))
    distance_sample = inference.approx.sample_node(distance, size=config['n_eval_samples'], more_replacements={ shared: var for shared, var in zip(approx_params, input_vars) })
    distance_mean = tt.mean(distance_sample)
    distance_function = theano.function(input_vars, distance_mean)
    distances = [distance_function(*[result[var][i] for var in inference.approx.shared_params.keys()]) for i in range(len(result['i']))]
    return pd.DataFrame({
        'date_time': result['date_time'],
        'error': np.stack(distances)
    })

def get_beast_scores(result_filename, config, true_pop_size):
    trace_df = pd.read_table(result_filename, parse_dates=['datetime'], comment='#')
    def get_score_for_iteration(i):
        to_use = trace_df[int(i * config['burn_in']):(i+1)]
        return np.mean(abs(to_use.popSize - true_pop_size))

    scores = [get_score_for_iteration(i) for i in tqdm.trange(trace_df.shape[0])]
    return pd.DataFrame({
        'date_time': trace_df['datetime'],
        'error': scores
    })

def get_mcmc_scores(trace, config, true_pop_size):
    draws = config['chain_length']
    log_every = config['log_every']
    pop_size_samples = trace.get_values('pop_size')[-draws:]
    def get_score_for_iteration(i):
        to_use = pop_size_samples[int(i * config['burn_in']):(i+1)]
        return np.mean(abs(to_use - true_pop_size))

    scores = [get_score_for_iteration(i) for i in range(0, draws, log_every)]
    return pd.DataFrame({
        'date_time': trace.times[-draws::log_every],
        'error': scores
    })

def get_variational_trace(result_filename, config, sequence_dict, newick_string):
    with open(result_filename, 'rb') as f:
        pymc_tracker = pickle.load(f)
    tree = newick.loads(newick_string)[0]
    model = variational_analysis.construct_model(config, tree, sequence_dict)
    inference = variational_analysis.construct_inference(config, model)
    for key, var in inference.approx.shared_params.items():
        var.set_value(pymc_tracker[key][-1])
    return inference.approx.sample(config['n_trace_samples'])

def get_trace_cols(config):
    return ['tree_height', 'kappa', 'pop_size'] + (['clock_rate'] if config['estimate_clock_rate'] else [])

def process_pymc_trace(trace, config, resample=False, burn_in=False):
    trace_df = pm.trace_to_dataframe(trace)
    if burn_in:
        trace_df = trace_df[int(trace_df.shape[0]*config['burn_in']):]
    root_height_col = 'tree__' + str(config['n_taxa'] - 2)
    full_trace = trace_df.rename({ root_height_col: 'tree_height' }, axis=1)[get_trace_cols(config)]

    if resample:
        return full_trace.sample(config['n_trace_samples'], random_state=config['seed'])
    else:
        return full_trace

def process_beast_trace(result_filename, config):
    trace_df = pd.read_table(result_filename, comment='#')
    to_use = trace_df[int(trace_df.shape[0]*config['burn_in']):]
    full_trace = to_use.rename({ 'TreeHeight': 'tree_height', 'popSize': 'pop_size', 'clockRate': 'clock_rate' }, axis=1)[get_trace_cols(config)]
    return full_trace.sample(config['n_trace_samples'], random_state=config['seed'])
