import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from pylo.common import GAP
from pymc3.distributions import Discrete
from pymc3 import Potential

# For stateless children, use dummy sequence
# For stateful children, use dummy child indices
def make_partial_probabilities(child_indices, child_transition_probs, child_sequences, child_leaf_mask):
	partial_probs = tt.alloc(0.0, child_indices.shape[0], child_sequences.shape[2], 4)
	
	def partials_from_partials(child_partials, child_transition_probs):
		# child_partials [child, site, child_char]
		child_partials_shuffled = child_partials.dimshuffle(0, 1, 'x', 2) # [child, site, (parent_char), child_char]
		transition_probs_shuffled = child_transition_probs.dimshuffle(0, 'x', 1, 2) # [child, (site), parent_char, child_char]
		return (child_partials_shuffled * transition_probs_shuffled).sum(axis=3)

	def partials_from_sequences(child_sequences, child_transition_probs):
		# child_transition_probs [child, parent_char, child_char] 
		children = tt.arange(child_sequences.shape[0]).dimshuffle(0, 'x') # [child, (site)]
		return tt.switch(tt.eq(child_sequences, GAP).dimshuffle(0, 1, 'x'), 1.0, child_transition_probs[children, :, child_sequences]) # [child, site, parent_char]

	def fill_row(node_index, child_indices, child_sequences, child_transition_probs, child_leaf_mask, partial_probs):
		leaf_child_partials = partials_from_sequences(child_sequences, child_transition_probs)
		node_child_partials = partials_from_partials(partial_probs[child_indices], child_transition_probs)
		return tt.set_subtensor(partial_probs[node_index], tt.switch(child_leaf_mask.dimshuffle(0, 'x', 'x'), leaf_child_partials, node_child_partials).prod(axis=0))
		
	partial_probs_filled = theano.scan(fill_row,
		sequences=[tt.arange(child_indices.shape[0]), child_indices, child_sequences, child_transition_probs, child_leaf_mask],
		outputs_info=partial_probs)[0][-1]

	return partial_probs_filled

def phylogenetic_log_likelihood(child_indices, child_transition_probs, child_patterns, child_leaf_mask, pattern_frequencies, character_frequencies):
    partials = make_partial_probabilities(child_indices, child_transition_probs, child_patterns, child_leaf_mask) # [node, site, char]
    root_partials = partials[-1] #[site, char]
    char_freqs_reshuffled = character_frequencies.dimshuffle('x', 0) #[(site), char]
    site_probs = (root_partials * char_freqs_reshuffled).sum(axis=1)
    return (tt.log(site_probs) * pattern_frequencies).sum()
    
def LeafSequences(name, topology, substitution_model, child_distances, child_patterns, pattern_frequencies, *args, **kwargs):
    transition_probs = substitution_model.get_transition_probs(child_distances)
    character_frequencies = substitution_model.get_equilibrium_probs()
    logp = phylogenetic_log_likelihood(
        tt.as_tensor_variable(topology.child_indices),
        transition_probs,
        child_patterns,
        tt.as_tensor_variable(topology.leaf_mask),
        pattern_frequencies,
        character_frequencies
    )
    return Potential(name, logp, *args, **kwargs)
        
