import numpy as np
import pandas as pd
import theano
import theano.tensor as tt

def get_variational_scores(result, config, model, inference, true_pop_size):
    approx_params = list(inference.approx.shared_params.values())
    distance = abs(model.pop_size - true_pop_size)/true_pop_size
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
        return np.mean(abs(to_use.popSize - true_pop_size))/true_pop_size

    scores = [get_score_for_iteration(i) for i in range(trace_df.shape[0])]
    return pd.DataFrame({
        'date_time': trace_df['datetime'],
        'error': scores
    })
