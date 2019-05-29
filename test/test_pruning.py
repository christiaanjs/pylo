from pylo.pruning import phylogenetic_log_likelihood
from pylo.transform import group_sequences
from pylo.hky import HKYSubstitutionModel
from pylo.topology import TreeTopology

import numpy as np
import theano.tensor as tt
import theano
from numpy.testing import assert_allclose

def test_pruning_value(taxa_encoded, tree):
    taxa_patterns, pattern_frequencies = group_sequences(taxa_encoded)
    pattern_frequencies = np.array(pattern_frequencies)

    topology = TreeTopology(tree)
    child_sequences = topology.build_sequence_table(taxa_patterns)
    node_heights = topology.get_init_heights()[topology.node_mask]
	
    kappa = 1.0
    pi = np.ones(4)/4
    
    kappa_ = tt.scalar()
    pi_ = tt.vector()
    node_heights_ = tt.vector()
    child_branch_lengths_ = topology.get_child_branch_lengths(node_heights_)
    child_sequences_ = tt.tensor3(dtype='int64')
    child_leaf_mask_ = tt.matrix(dtype='bool')
    child_indices_ = tt.matrix(dtype='int64')
    pattern_frequencies_ = tt.vector(dtype='int64')

    substitution_model = HKYSubstitutionModel(kappa_, pi_)
    child_transition_probs_ = substitution_model.get_transition_probs(child_branch_lengths_)
	
    ll_ = phylogenetic_log_likelihood(child_indices_, child_transition_probs_, child_sequences_, child_leaf_mask_, pattern_frequencies_, pi_)
    
    f = theano.function([kappa_, pi_, node_heights_, child_indices_, child_sequences_, child_leaf_mask_, pattern_frequencies_], ll_)
    res = f(kappa, pi, node_heights, topology.node_child_indices, child_sequences, topology.get_node_child_leaf_mask(), pattern_frequencies)
    
    assert_allclose(res, -1992.2056440317247)


def test_pruning_gradient_substmodel(taxa_encoded, tree):
    taxa_patterns, pattern_frequencies = group_sequences(taxa_encoded)
    pattern_frequencies = np.array(pattern_frequencies)

    dummy_seq = np.repeat(-1, len(list(taxa_patterns.values())[0]))
    child_branch_lengths, child_sequences, child_leaf_mask, child_indices = get_tables_np(tree, taxa_patterns, dummy_seq)
	
    kappa = 1.0
    pi = np.ones(4)/4
    
    kappa_ = tt.scalar()
    pi_ = tt.vector()
    child_branch_lengths_ = tt.matrix()
    child_sequences_ = tt.tensor3(dtype='int64')
    child_leaf_mask_ = tt.matrix(dtype='bool')
    child_indices_ = tt.matrix(dtype='int64')
    pattern_frequencies_ = tt.vector(dtype='int64')

    child_transition_probs_ = hky_transition_probs_mat(kappa_, pi_, child_branch_lengths_)
	
    ll_ = phylogenetic_log_likelihood(child_indices_, child_transition_probs_, child_sequences_, child_leaf_mask_, pattern_frequencies_, pi_)
    
    f = theano.function([kappa_, pi_, child_branch_lengths_, child_indices_, child_sequences_, child_leaf_mask_, pattern_frequencies_], ll_)

    grad_ = tt.grad(ll_, pi_)
    grad_f = theano.function([kappa_, pi_, child_branch_lengths_, child_indices_, child_sequences_, child_leaf_mask_, pattern_frequencies_], grad_)
    grad_theano = grad_f(kappa, pi, child_branch_lengths, child_indices, child_sequences, child_leaf_mask, pattern_frequencies)

    partial_f = lambda pi: f(kappa, pi, child_branch_lengths, child_indices, child_sequences, child_leaf_mask, pattern_frequencies)

    grad_numeric = nd.Gradient(partial_f)(pi)
    
    assert_allclose(grad_theano, grad_numeric, rtol = GRAD_RTOL)

	
def test_pruning_gradient_branch(taxa_encoded, tree):
    taxa_patterns, pattern_frequencies = group_sequences(taxa_encoded)
    pattern_frequencies = np.array(pattern_frequencies)

    dummy_seq = np.repeat(-1, len(list(taxa_patterns.values())[0]))
    child_branch_lengths, child_sequences, child_leaf_mask, child_indices = get_tables_np(tree, taxa_patterns, dummy_seq)
	
    kappa = 1.0
    pi = np.ones(4)/4
    
    kappa_ = tt.scalar()
    pi_ = tt.vector()
    child_branch_lengths_ = tt.matrix()
    child_sequences_ = tt.tensor3(dtype='int64')
    child_leaf_mask_ = tt.matrix(dtype='bool')
    child_indices_ = tt.matrix(dtype='int64')
    pattern_frequencies_ = tt.vector(dtype='int64')

    child_transition_probs_ = hky_transition_probs_mat(kappa_, pi_, child_branch_lengths_)
	
    ll_ = phylogenetic_log_likelihood(child_indices_, child_transition_probs_, child_sequences_, child_leaf_mask_, pattern_frequencies_, pi_)
    
    f = theano.function([kappa_, pi_, child_branch_lengths_, child_indices_, child_sequences_, child_leaf_mask_, pattern_frequencies_], ll_)

    grad_ = tt.grad(ll_, child_branch_lengths_)
    grad_f = theano.function([kappa_, pi_, child_branch_lengths_, child_indices_, child_sequences_, child_leaf_mask_, pattern_frequencies_], grad_)
    grad_theano = grad_f(kappa, pi, child_branch_lengths, child_indices, child_sequences, child_leaf_mask, pattern_frequencies)

    branch_lengths_shape = child_branch_lengths.shape
    partial_f = lambda branch_lengths: f(kappa, pi, branch_lengths.reshape(branch_lengths_shape), child_indices, child_sequences, child_leaf_mask, pattern_frequencies)

    grad_numeric = nd.Gradient(partial_f)(child_branch_lengths.flatten())
    
    assert_allclose(grad_theano, grad_numeric.reshape(branch_lengths_shape), rtol = GRAD_RTOL)

