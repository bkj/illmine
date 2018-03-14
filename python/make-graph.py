
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pprint
from time import time

# --
# Create graph

def make_graph(edges):
    g = nx.Graph()
    
    for i, row in edges.iterrows():
        g.add_node(row.src)
        g.add_node(row.trg)
        g.add_edge(row.src, row.trg, weight=row.weight, type=tuple(sorted((row.src_type, row.trg_type))), assigned=0)
        
        g.nodes[row.src]['_type'] = row.src_type
        g.nodes[row.trg]['_type'] = row.trg_type
    
    return g


def compute_signatures(g):
    for node in g.nodes:
        weights = {}
        counter = defaultdict(set)
        one_hop = set(g.neighbors(node))
        for neib in one_hop:
            
            # One hops
            _type = g.nodes[neib]['_type']
            weight = g.edges[(node, neib)]['weight']
            
            if weight > weights.get(_type, -1):
                weights[_type] = weight
            
            counter[_type].add(neib)
            
            # Two hops
            for neib2 in g.neighbors(neib):
                if (neib2 != node) and (neib2 not in one_hop):
                    _type2 = g.nodes[neib2]['_type']
                    weight2 = g.edges[(neib, neib2)]['weight']
                    
                    if weight + weight2 > weights.get((_type, _type2), -1):
                        weights[(_type, _type2)] = weight + weight2
                    
                    counter[(_type, _type2)].add(neib2)
            
            g.nodes[node]['weights'] = weights
            g.nodes[node]['counter'] = dict([(k, len(v)) for k,v in counter.items()])
    
    return g

# --
# Load query

query = pd.read_csv('../queries/queryGraph.Path.2.4.txt', sep='#', skiprows=5, header=None)
query.columns = ('src', 'trg', 'weight')
query = query[query.src <= query.trg].reset_index(drop=True)
query.src -= 1
query.trg -= 1
query.weight = 0

query_types = pd.read_csv('../queries/queryTypes.Path.2.4.txt', sep='\t', header=None)
query_types.columns = ('node_id', 'type')
qtype_lookup = pd.Series(np.array(query_types.type).astype(int), index=np.array(query_types.node_id) - 1).to_dict()

query['src_type']  = query.src.apply(lambda x: qtype_lookup[x])
query['trg_type']  = query.trg.apply(lambda x: qtype_lookup[x])
query['edge_type'] = query[['src_type', 'trg_type']].apply(lambda x: '%d#%d' % tuple(sorted((x['src_type'], x['trg_type']))), axis=1)

q_types = set(qtype_lookup.values())
q = make_graph(query)
q = compute_signatures(q)

# --
# Load graph

edges = pd.read_csv('./data/edges.tsv', sep='\t')
edges = edges[edges.edge_type.isin(query.edge_type)]
edges = edges.sort_values(['weight', 'src', 'trg'], ascending=[False, True, True]).reset_index(drop=True)
edges.weight = edges.weight.apply(lambda x: '%0.1f' % x).astype(float)

g = make_graph(edges)
g = compute_signatures(g)

assert len(g.edges) == edges.shape[0]

edges[edges.src == 836]

# --
# Prune graph

# Compute bounds on query counts

pd.DataFrame([q.nodes[n] for n in q.nodes]).groupby('_type').counter.apply(lambda x: x.keys())

count_bound_types = {}
for node in q.nodes:
    node_type = q.nodes[node]['_type']
    node_counter = q.nodes[node]['counter']
    
    if not node_type in count_bound_types:
        count_bound_types[node_type] = set(node_counter.keys())
    else:
        count_bound_types[node_type] = count_bound_types[node_type].intersection(node_counter.keys())

count_bounds = defaultdict(dict)
for node in q.nodes:
    node_type = q.nodes[node]['_type']
    node_counter = q.nodes[node]['counter']
    
    for count_type, count in node_counter.items():
        if count_type in count_bound_types[node_type]:
            if count_bounds[node_type].get(count_type, np.inf) > count:
                count_bounds[node_type][count_type] = count


# Prune graph w/ bounds
g_nodes = list(g.nodes)
for node in g_nodes:
    node_type = g.nodes[node]['_type']
    node_counter = g.nodes[node]['counter']
    
    type_bounds = count_bounds[node_type]
    for count_type, count in type_bounds.items():
        if node_counter.get(count_type, -1) < count:
            g.remove_node(node)
            break

sel = edges[['src', 'trg']].apply(lambda x: g.has_edge(x['src'], x['trg']), axis=1)
edges = edges[sel].reset_index(drop=True)

assert len(g.edges) == edges.shape[0]

# >>

edges[(edges.src == 399) & (edges.trg == 908)]
edges[(edges.src == 203) & (edges.trg == 836)]
edges[(edges.src == 836) & (edges.trg == 908)]

# <<

# --
# Run

import copy
import heapq

num_query_nodes = len(q.nodes)
num_query_edges = len(q.edges)

def init_candidates(q, edge, _type):
    for q_edge_idx in q.edges:
        q_edge = q.edges[q_edge_idx]
        if (q_edge['type'] == _type) and (not q_edge.get('assigned')):
            c = q.copy()
            c.edges[q_edge_idx].update(edge)
            c.nodes[q_edge_idx[0]]['align'] = edge['src']
            c.nodes[q_edge_idx[1]]['align'] = edge['trg']
            yield c


class Pruner(object):
    def __init__(self, n=20):
        self.n = n
        self.buffer = []
        self.counter = 0
    
    def keep(self, x):
        self.counter += 1
        
        weight = sum([xx[2] for xx in x])
        if len(self.buffer) < self.n:
            heapq.heappush(self.buffer, (weight, x))
            return True
        else:
            if weight > self.buffer[0][0]:
                heapq.heapreplace(self.buffer, (weight, x))
                return True
            else:
                return False
    
    @property
    def min(self):
        if len(self.buffer) == self.n:
            return self.buffer[0][0]
        else:
            return -np.inf


def process_row(row_edge, row_type):
    for initial_candidate in init_candidates(q, row_edge, row_type):
        candidates = expand_graphs(g, initial_candidate, max_weight=row_edge['weight'])
        for candidate in filter(pruner.keep, set(candidates)):
            pass


def expand_graphs(g, candidate, max_weight):
    aligned_nodes = [n for n in candidate.nodes if 'align' in candidate.nodes[n]]
    used_nodes = [candidate.nodes[n]['align'] for n in aligned_nodes]
    
    # if (399 in used_nodes) and (908 in used_nodes) and (836 in used_nodes):
    #     flag = True
    #     print('--', [candidate.edges[e] for e in candidate.edges])
    # else:
    flag = False
    
    num_unaligned_nodes = len(candidate.nodes) - len(aligned_nodes)
    
    if num_unaligned_nodes == 0:
        candidate_edges = [candidate.edges[e] for e in candidate.edges]
        
        if flag:
            print('*****', candidate_edges)
        
        yield tuple([(t['src'], t['trg'], t['weight']) if t['assigned'] else None for t in candidate_edges])
    else:
        for aligned_node in aligned_nodes: # Node in query
            
            adjacent_edge_qidxs  = nx.edges(candidate, aligned_node) # Edges in query
            potential_edge_gidxs = nx.edges(g, candidate.nodes[aligned_node]['align']) # Edges in graph
            
            if flag:
                print(candidate.nodes[aligned_node]['align'], potential_edge_gidxs)
            
            for adjacent_edge_qidx in adjacent_edge_qidxs:
                adjacent_edge = candidate.edges[adjacent_edge_qidx]
                if not adjacent_edge['assigned']:
                    for potential_edge_gidx in potential_edge_gidxs:
                        
                        if flag and 203 in potential_edge_gidx:
                                print('+')
                        
                        if (potential_edge_gidx[1] not in used_nodes):
                            potential_edge = g.edges[potential_edge_gidx]
                            
                            if flag and 203 in potential_edge_gidx:
                                    print('++', potential_edge['type'], adjacent_edge['type'])
                            
                            if potential_edge['type'] == adjacent_edge['type']:
                                
                                if flag and 203 in potential_edge_gidx:
                                        print('+++', potential_edge_gidx, potential_edge)
                                
                                if potential_edge_gidx not in done:
                                    if flag and 203 in potential_edge_gidx:
                                            print('++++')

                                    potential_weight = potential_edge['weight']
                                    upper_bound = candidate.size(weight='weight') + potential_weight + max_weight * (num_unaligned_nodes - 1)
                                    if upper_bound > pruner.min:
                                        
                                        c = candidate.copy()
                                        c.edges[adjacent_edge_qidx].update({
                                            "assigned" : True,
                                            "src"      : potential_edge_gidx[0],
                                            "trg"      : potential_edge_gidx[1],
                                            "weight"   : potential_weight,
                                        })
                                        
                                        c.nodes[adjacent_edge_qidx[0]]['align'] = potential_edge_gidx[0]
                                        c.nodes[adjacent_edge_qidx[1]]['align'] = potential_edge_gidx[1]
                                        
                                        for cc in expand_graphs(g, c, max_weight):
                                            yield cc


# !! There's some minor bug here -- not sure where
# 986#33#0.9618458813394116   234#393#0.9523722094624016  33#234#0.92 2.834218090801813
# 

pruner = Pruner(n=200)
done = set([])

for e in g.edges:
    g.edges[e]['done'] = False

t = time()

for i, row in edges.iterrows():
    src, trg, weight, src_type, trg_type = row.src, row.trg, row.weight, row.src_type, row.trg_type
    
    if (weight * num_query_edges) < pruner.min:
        break
    
    row_type = tuple(sorted((src_type, trg_type)))
    
    process_row({"assigned" : True, "src" : src, "trg" : trg, "weight" : weight}, row_type)
    done.add((src, trg))
    
    if src_type == trg_type:
        process_row({"assigned" : True, "src" : trg, "trg" : src, "weight" : weight}, row_type)
        done.add((trg, src))

for score, obj in sorted(pruner.buffer, key=lambda x: x[0]):
    print('%0.2f\t%s' % (score, obj))

print(sum([p[0] for p in pruner.buffer]))
print(pruner.counter)
print(time() - t)
print(i)