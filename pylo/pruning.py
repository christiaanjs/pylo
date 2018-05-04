import theano
import theano.tensor as tt
from theano.ifelse import ifelse

from pylo.common import GAP

# For stateless children, use dummy sequence
# For stateful children, use dummy child indices
def phylogenetic_log_likelihood(child_indices, child_transition_probs, child_sequences, child_leaf_mask):
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
		outputs_info=partial_probs)[0]

	return partial_probs_filled
				
	
