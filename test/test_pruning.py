from pylo.pruning import phylogenetic_log_likelihood
from pylo.transform import get_tables
from pylo.hky import hky_transition_probs_mat

import numpy as np
import theano.tensor as tt
import theano
from numpy.testing import assert_allclose

def test_pruning(taxa_encoded, tree):
    dummy_seq = np.repeat(-1, len(list(taxa_encoded.values())[0]))
    child_branch_lengths, child_sequences, child_leaf_mask, child_indices = [np.array(x) for x in get_tables(tree, taxa_encoded, dummy_seq)]
	
    kappa = 1.0
    pi = np.ones(4)/4
    
    kappa_ = tt.scalar()
    pi_ = tt.vector()
    child_branch_lengths_ = tt.matrix()
    child_sequences_ = tt.tensor3(dtype='int64')
    child_leaf_mask_ = tt.matrix(dtype='bool')
    child_indices_ = tt.matrix(dtype='int64')

    child_transition_probs_ = hky_transition_probs_mat(kappa_, pi_, child_branch_lengths_)
	
    ll_ = phylogenetic_log_likelihood(child_indices_, child_transition_probs_, child_sequences_, child_leaf_mask_)
    
    f = theano.function([kappa_, pi_, child_branch_lengths_, child_indices_, child_sequences_, child_leaf_mask_], ll_)
    res = f(kappa, pi, child_branch_lengths, child_indices, child_sequences, child_leaf_mask)
    
    assert_allclose(res, -1992.2056440317247)



		
	
		
