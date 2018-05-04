from pylo.common import DUMMY_INDEX

def list_concat(lists, last_item):
    return [y for x in lists for y in x] + [last_item]

def process_child_result(child, child_result, taxa_dict, dummy_seq):
    branch_lengths, _, _, _ = child_result
    child_leaf = len(branch_lengths) == 0
    if child_leaf:
        return child.length, taxa_dict[child.name], True
    else:
        return child.length, dummy_seq, False

def reverse_cumsum(x):
    sum_ = 0
    res = [0]*len(x)
    res[-1] = x[-1]
    for i in range(len(x) - 2, -1, -1):
        res[i] = x[i] + res[i + 1]
    return res
    
def get_tables_offset(node, taxa_dict, dummy_seq):
    children = node.descendants
    if(len(children) == 0):
        return [], [], [], []
    else:
        child_results = [get_tables_offset(child, taxa_dict, dummy_seq) for child in children]
        branch_length_results, sequence_results, leaf_results, offset_results = zip(*child_results)
        
        branch_lengths, sequences, leaf_mask = zip(*[process_child_result(child, result, taxa_dict, dummy_seq) for child, result in zip(children, child_results)])
        child_offsets = [DUMMY_INDEX if is_leaf else x for x, is_leaf in zip(reverse_cumsum([len(x) for x in offset_results[1:]] + [1]), leaf_mask)]

        return (list_concat(branch_length_results, list(branch_lengths)),
            list_concat(sequence_results, list(sequences)),
            list_concat(leaf_results, list(leaf_mask)),
            list_concat(offset_results, child_offsets))

def get_tables(node, taxa_dict, dummy_seq):
    branch_lengths, sequences, leaf_mask, offsets = get_tables_offset(node, taxa_dict, dummy_seq)
    indices = [ [DUMMY_INDEX if is_leaf else i - offset for offset, is_leaf in zip(offset_row, leaf_row) ] for i, (offset_row, leaf_row) in enumerate(zip(offsets, leaf_mask)) ]
    return branch_lengths, sequences, leaf_mask, indices
