from pylo.transform import get_parent_indices, get_tables_np
import numpy as np
import theano
import theano.tensor as tt

class TreeTopology(object):
    def __init__(self, tree):
        _, self.leaf_children, self.leaf_mask, self.child_indices = get_tables_np(tree)
        self.parent_indices = np.array(get_parent_indices(self.child_indices))
    
    def build_sequence_table(self, sequence_dict, dummy_seq=None):
        if dummy_seq is None:
            dummy_seq = list(sequence_dict.values())[0]
        
        return [[dummy_seq if name is None else sequence_dict[name] for name in node] for node in self.leaf_children]

    def get_heights(self, root_height, non_root_proportions):
        n_internal_nodes = len(self.leaf_children)
        parent_indices_reversed = self.parent_indices[::-1]
        parent_indices_inorder = np.where(parent_indices_reversed == -1, -1, n_internal_nodes - parent_indices_reversed - 1)
        parent_proportions = tt.concatenate([non_root_proportions, tt.ones(1)])
        parent_proportions_inorder = parent_proportions[::-1]
        func = lambda i, parent, prop, out: tt.set_subtensor(out[i], tt.where(tt.eq(parent, -1), prop, prop*out[parent]))
        ixs = tt.arange(n_internal_nodes)
        out_init = tt.zeros(n_internal_nodes)
        heights_inorder = theano.scan(
            func,
            sequences=(ixs, tt.as_tensor(parent_indices_inorder), parent_proportions_inorder),
            outputs_info=out_init)[0][-1]
        return root_height*heights_inorder[::-1] 

    def get_proportions(self, heights):
        root_height = heights[-1]
        non_root_heights = heights[:-1]
        non_root_parent_indices = self.parent_indices[:-1]
        parent_heights = heights[non_root_parent_indices]
        return non_root_heights/parent_heights, root_height


    def get_child_branch_lengths(self, heights):
        child_heights = tt.where(tt.eq(self.child_indices, -1), 0.0, heights[self.child_indices])
        return heights.dimshuffle(0, 'x') - child_heights


    def get_internal_node_count(self):
        return len(self.leaf_children)

    def get_taxon_count(self):
        return self.get_internal_node_count() + 1

    def get_root_index(self):
        return self.get_internal_node_count() - 1
            
