import numpy as np
from variational_analysis import construct_model, construct_inference
import theano
import theano.tensor as tt

def get_variational_scores(result, config):
    model = construct_model(config)
    inference = construct_inference(config, model)
    approx_params = list(inference.approx.shared_params.values())
    true_pop_size = config['pop_size']
    distance = abs(model.pop_size - true_pop_size)/true_pop_size
    input_vars = tt.dvectors(len(approx_params))
    distance_sample = inference.approx.sample_node(distance, size=config['n_eval_samples'], more_replacements={ shared: var for shared, var in zip(approx_params, input_vars) })
    distance_mean = tt.mean(distance_sample)
    distance_function = theano.function(input_vars, distance_mean)
    distances = [distance_function(*[result[var][i] for var in inference.approx.shared_params.keys()]) for i in range(len(result['i']))]
    return np.stack(distances)
