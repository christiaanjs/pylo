from pylo.common import *
import pandas as pd
import numpy as np

def list_concat(lists, last_item):
    return [y for x in lists for y in x] + [last_item]

def encode_sequences(taxa_dict):
    state_dict = { 'A': A, 'C': C, 'G': G, 'T': T, '-': GAP }
    return { name: [state_dict[char] for char in sequence] for name, sequence in taxa_dict.items() }

def get_dummy_seq(taxa_dict):
    return np.repeat(GAP, len(list(taxa_dict.values())[0])) 

def process_child_result(child, child_result):
    branch_lengths, _, _, _ = child_result
    child_leaf = len(branch_lengths) == 0
    if child_leaf:
        return child.length, child.name, True
    else:
        return child.length, None, False

def group_sequences(taxa_dict):
    taxon_names = list(taxa_dict.keys())
    taxa_df = pd.DataFrame(taxa_dict)
    pattern_series = taxa_df.groupby(taxon_names).size()
    pattern_dict = pattern_series.index.to_frame().to_dict(orient='list')
    return pattern_dict, pattern_series.values

def reverse_cumsum(x):
    sum_ = 0
    res = [0]*len(x)
    res[-1] = x[-1]
    for i in range(len(x) - 2, -1, -1):
        res[i] = x[i] + res[i + 1]
    return res
    
def get_tables_offset(node):
    children = node.descendants
    if(len(children) == 0):
        return [], [], [], []
    else:
        child_results = [get_tables_offset(child) for child in children]
        branch_length_results, leaf_children_results, leaf_results, offset_results = zip(*child_results)
        
        branch_lengths, leaf_children, leaf_mask = zip(*[process_child_result(child, result) for child, result in zip(children, child_results)])
        child_offsets = [DUMMY_INDEX if is_leaf else x for x, is_leaf in zip(reverse_cumsum([len(x) for x in offset_results[1:]] + [1]), leaf_mask)]

        return (list_concat(branch_length_results, list(branch_lengths)),
            list_concat(leaf_children_results, list(leaf_children)),
            list_concat(leaf_results, list(leaf_mask)),
            list_concat(offset_results, child_offsets))

def get_tables(node):
    branch_lengths, leaf_children, leaf_mask, offsets = get_tables_offset(node)
    indices = [ [DUMMY_INDEX if is_leaf else i - offset for offset, is_leaf in zip(offset_row, leaf_row) ] for i, (offset_row, leaf_row) in enumerate(zip(offsets, leaf_mask)) ]
    return branch_lengths, leaf_children, leaf_mask, indices

def get_tables_np(node):
    return tuple([np.array(x) for x in get_tables(node)])

def get_parent_indices(child_indices):
    n_internal_nodes = len(child_indices)
    parent_indices = [-1]*n_internal_nodes
    for i in range(n_internal_nodes):
        for child_index in child_indices[i]:
            if child_index != -1:
                parent_indices[child_index] = i
    return parent_indices

def get_node_heights(node):
    if node.is_leaf:
        return []
    else:
        child_heights = [get_node_heights(child) for child in node.descendants]
        child_branch_lengths = [child.length for child in node.descendants]
        node_height = max([single_child_heights[-1] + branch_length if len(single_child_heights) > 0 else branch_length for single_child_heights, branch_length in zip(child_heights, child_branch_lengths)])
        return np.concatenate(child_heights + [[node_height]])
        
