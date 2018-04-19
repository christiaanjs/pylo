import theano
import theano.tensor as tt
from theano.ifelse import ifelse

def phylogenetic_likelihood_math(child_indices, child_transition_probs, leaf_parent_mask, leaf_sequences):
	partial_probs = tt.alloc(0.0, child_indices.shape[0], leaf_sequences.shape[1], 4)
	def get_row_leaf_parent(child_indices, child_transition_probs, leaf_sequences):
		# child_transition_probs [child, parent_char, child_char] 
		child_sequences = leaf_sequences[child_indices] # [child, site]
		children = tt.arange(child_indices.shape[0]).dimshuffle(0, 'x') # [child, (site)]
		sequence_transition_probs = child_transition_probs[children, :, child_sequences] # [child, site, parent_char]
		return sequence_transition_probs.prod(axis=0)
		
	def get_row_non_leaf_parent(child_indices, child_transition_probs, partial_probs):
		child_partials = partial_probs[child_indices] # [child, site, child_char]
		child_partials_shuffled = child_partials.dimshuffle(1, 'x', 0, 2) # [site, (parent_char), child, child_char]
		transition_probs_shuffled = child_transition_probs.dimshuffle('x', 1, 0, 2) # [(site), parent_char, child, child_char]
		return (child_partials_shuffled * transition_probs_shuffled).sum(axis=3).prod(axis=2)

	def fill_row(node_index, child_indices, child_transition_probs, is_leaf_parent,	partial_probs, leaf_sequences):
		return tt.set_subtensor(partial_probs[node_index], ifelse(is_leaf_parent,
			get_row_leaf_parent(child_indices, child_transition_probs, leaf_sequences),
			get_row_non_leaf_parent(child_indices, child_transition_probs, partial_probs)))
		
	return theano.scan(fill_row,
		sequences=[tt.arange(child_indices.shape[0]), child_indices, child_transition_probs, leaf_parent_mask],
		outputs_info=partial_probs,
		non_sequences=[leaf_sequences])
				
	
