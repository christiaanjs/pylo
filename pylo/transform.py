def list_concat(lists, last_item):
    return [y for x in lists for y in x] + [last_item]

def process_child_result(child, child_result, taxa_dict, dummy_seq):
    branch_lengths, _, _ = child_result
    child_leaf = len(branch_lengths) == 0
    if child_leaf:
        return child.length, taxa_dict[child.name], True
    else:
        return child.length, dummy_seq, False
    
def get_tables(node, taxa_dict, dummy_seq):
    children = node.descendants
    if(len(children) == 0):
        return [], [], []
    else:
        child_results = [get_tables(child, taxa_dict, dummy_seq) for child in children]
        branch_length_results, sequence_results, leaf_results = zip(*child_results)
        
        branch_lengths, sequences, leaf_mask = zip(*[process_child_result(child, result, taxa_dict, dummy_seq) for child, result in zip(children, child_results)])

        return (list_concat(branch_length_results, list(branch_lengths)),
            list_concat(sequence_results, list(sequences)),
            list_concat(leaf_results, list(leaf_mask)))

