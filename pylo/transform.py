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

def group_sequences(taxa_dict):
    taxon_names = list(taxa_dict.keys())
    taxa_df = pd.DataFrame(taxa_dict)
    pattern_series = taxa_df.groupby(taxon_names).size()
    pattern_dict = pattern_series.index.to_frame().to_dict(orient='list')
    return pattern_dict, pattern_series.values

