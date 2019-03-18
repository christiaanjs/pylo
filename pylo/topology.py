import numpy as np
import theano
import theano.tensor as tt

def traverse_postorder(node, leaf_f, node_f, transform=False):
    if node.is_leaf:
        return np.array([leaf_f(node)])
    else:
        child_results = [traverse_postorder(child, leaf_f, node_f, transform=transform) for child in node.descendants]
        if transform:
            node_result, child_results = node_f(node, child_results) 
        else:
            node_result = node_f(node, child_results)
        return np.concatenate(child_results + [[node_result]])

def get_heights(node):
    leaf_f = lambda x: 0.0
    def node_f(node, child_heights):
        child_branch_lengths = [child.length for child in node.descendants]
        child_branch_heights = [single_child_heights[-1] + branch_length for single_child_heights, branch_length in zip(child_heights, child_branch_lengths)]
        node_height = max(child_branch_heights)
        adjusted_child_heights = [single_child_heights + (node_height - child_branch_height) for single_child_heights, child_branch_height in zip(child_heights, child_branch_heights)]
        return node_height, adjusted_child_heights
    return traverse_postorder(node, leaf_f, node_f, transform=True)

def get_max_leaf_descendant_heights(node):
    leaf_f = lambda x: [0.0, 0.0]
    def node_f(node, child_results):
        child_branch_lengths = [child.length for child in node.descendants]
        child_branch_heights = [single_child_results[-1, 0] + branch_length for single_child_results, branch_length in zip(child_results, child_branch_lengths)]
        node_height = max(child_branch_heights)
        adjusted_child_results = [single_child_results + (node_height - child_branch_height) for single_child_results, child_branch_height in zip(child_results, child_branch_heights)]
        max_leaf_descendant_height = max(adjusted_result[-1, 1] for adjusted_result in adjusted_child_results)
        return [node_height, max_leaf_descendant_height], adjusted_child_results
    return traverse_postorder(node, leaf_f, node_f, transform=True)[:, 1]

def get_leaf_mask(node):
    return traverse_postorder(node, lambda x: True, lambda x, y: False)

def get_parent_indices(node):
    leaf_f = lambda x: -1
    def node_f(node, child_results):
        lengths = np.array([len(child_result) for child_result in child_results])
        node_index = np.sum(lengths)
        adjustments = np.concatenate(([0], np.cumsum(lengths[:-1])))
        adjusted_child_results = [np.concatenate((child_result[:-1] + adjustment, [node_index])) for child_result, adjustment in zip(child_results, adjustments)]

        return -1, adjusted_child_results
    return traverse_postorder(node, leaf_f, node_f, transform=True)

def get_child_indices(node): # Assumes binary trees
    leaf_f = lambda x: [-1, -1]
    def node_f(node, child_results):
        lengths = np.array([len(child_result) for child_result in child_results])
        adjustments = np.concatenate(([0], np.cumsum(lengths[:-1])))
        node_children = np.cumsum(lengths) - 1 
        adjusted_child_results = [np.where(child_result == -1, -1, child_result + adjustment) for child_result, adjustment in zip(child_results, adjustments)]
        return node_children, adjusted_child_results
    return traverse_postorder(node, leaf_f, node_f, transform=True)

def get_names(node):
    leaf_f = lambda x: x.name
    node_f = lambda x, y: None
    return traverse_postorder(node, leaf_f, node_f)

class TreeTopology(object):
    def __init__(self, tree):
        self.tree = tree
        self.names = get_names(tree)
        self.leaf_mask = get_leaf_mask(tree)
        self.node_mask = np.logical_not(self.leaf_mask)
        self.child_indices = get_child_indices(tree)
        self.parent_indices = get_parent_indices(tree)
        self.max_leaf_descendant_heights = get_max_leaf_descendant_heights(tree)
        self.node_indices = self.node_mask.nonzero()
        self.node_index_mapping = (np.arange(len(self.names))[:, np.newaxis] == self.node_indices).argmax(axis=1)

    def get_init_heights(self):
        return get_heights(self.tree)
    
    def build_sequence_table(self, sequence_dict, dummy_seq=None):
        if dummy_seq is None:
            dummy_seq = list(sequence_dict.values())[0]
               
        node_child_indices = self.child_indices[self.leaf_mask]
        leaf_child_names = np.where(self.leaf_mask[node_child_indices], self.names[node_child_indices], None)        

        return [[dummy_seq if name is None else sequence_dict[name] for name in node] for node in leaf_child_names]

    def get_proportions(self, heights):
        root_height = heights[-1]
        non_root_heights = heights[:-1]
        non_root_parent_indices = self.parent_indices[self.node_mask][:-1]
        parent_heights = heights[self.node_index_mapping[non_root_parent_indices]]
        min_heights = self.max_leaf_descendant_heights[self.node_mask][:-1]
        return (non_root_heights - min_heights)/(parent_heights - min_heights), root_height

    def get_heights(self, root_height, proportions):
        pass # TODO

    def get_child_branch_lengths(self, heights):
        child_heights = heights[self.child_indices[self.node_mask]]
        return heights.dimshuffle(0, 'x') - child_heights

    def get_internal_node_count(self):
        return np.sum(self.node_mask)

    def get_taxon_count(self):
        return np.sum(self.leaf_mask)

    def get_root_index(self):
        return self.get_internal_node_count()  - 1

