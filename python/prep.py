#!/usr/bin/env python

"""
    prep.py
"""

from __future__ import print_function

import sys
import numpy as np
import pandas as pd

# --
# IO (actual code)

# edges = pd.read_csv('../sample_1000_10000.txt', sep=' ', skiprows=8, header=None)
# edges = edges[[1,2,3]]
# edges.columns = ('src', 'trg', 'weight')
# edges.weight /= 100

# edges[['src', 'trg']] = edges[['src', 'trg']].apply(lambda x: sorted((x['src'], x['trg'])), axis=1)
# edges = edges.drop_duplicates(['src', 'trg'])

# !! Some nodes don't have any edges, so UIUC randomly adds them.

# --
# IO (compatibility mode)

edges = pd.read_csv('../GT_1000_10000.txt', sep='#', skiprows=5, header=None)
edges.columns = ('src', 'trg', 'weight')
edges = edges.sort_values(['src', 'trg'])

edges = edges[edges.src < edges.trg].reset_index(drop=True)

# --
# Add node labels

node_types = pd.read_csv('../types_1000_10000.txt', sep='\t', header=None)
node_types.columns = ('node_id', 'type')

edges = pd.merge(edges, node_types, left_on='src', right_on='node_id')
edges.columns = ('src', 'trg', 'weight', 'delete_me_1', 'src_type')

edges = pd.merge(edges, node_types, left_on='trg', right_on='node_id')
edges.columns = ('src', 'trg', 'weight', 'delete_me_1', 'src_type', 'delete_me_2', 'trg_type')

del edges['delete_me_1']
del edges['delete_me_2']

edges['edge_type'] = edges[['src_type', 'trg_type']].apply(lambda x: '%d#%d' % tuple(sorted((x['src_type'], x['trg_type']))), axis=1)

# --
# Write to disk

edges = edges.sort_values('weight', ascending=False).reset_index(drop=True)

edges.to_csv('./data/edges.tsv', sep='\t', index=False)

for edge_type in np.unique(edges.edge_type):
    print('saving %s' % edge_type, file=sys.stderr)
    tmp = edges[edges.edge_type == edge_type]
    tmp.to_csv('./data/GT_1000_10000_%s.list' % edge_type, sep='\t', header=None, index=False)

