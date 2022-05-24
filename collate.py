import random
import numpy as np
import networkx as nx

PAD = '<PAD>'
NXT, PRE = '<NEXT>', '<RREVIOUS>'
SEL, NPL = '<SELF_LOOP>', '<NEXT_PREVIOUS_LOOP>'
CLS, RCLS = '<CLS>', '<RCLS>'
TL = '<TOO_LONG>'

class Vocab(object):
    def __init__(self, vocabs):
        idx2token = vocabs
        self._idx2token = idx2token
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._padding_idx = self._token2idx[PAD]
    
    @property
    def size(self):
        return len(self._idx2token)

    @property
    def padding_idx(self):
        return self._padding_idx
    
    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token[i] for i in x]
        return self._idx2token[x]
    
    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx[x]

relation_vocab = Vocab([PAD, NXT, PRE, SEL, NPL, CLS, RCLS, TL])

class SeqGraph(object):
    def __init__(self, seq):
        self.seq = seq
        self.graph, self.root, self.nid2iid = self.seq_to_graph(seq)

    def seq_to_graph(self, seq):
        nodes = np.unique(seq)
        nid2iid = {i: nodes[i] for i in range(len(nodes))}
        root = np.where(nodes == seq[0])[0][0]
        g = nx.DiGraph()
        for i in range(len(seq) - 1):
            v = np.where(nodes == seq[i])[0][0]
            if seq[i+1] == 0:
                break
            u = np.where(nodes == seq[i+1])[0][0]
            if u == v:
                continue
            else:
                if g.has_edge(u, v):
                    if g[u][v]['label'] == PRE:
                        g.add_edge(u, v, label=NPL)
                        g.add_edge(v, u, label=NPL)
                else:
                    g.add_edge(u, v, label=NXT)
                    g.add_edge(v, u, label=PRE)
        for u, node in enumerate(nodes):
            # if((u, u)) not in g.edges:
            #     g.add_node(u)
            #     g.add_edge(u, u, label=SEL)
            g.add_edge(u, u, label=SEL)
        return g, root, nid2iid

    def bfs(self):
        g = self.graph
        queue = [self.root]
        depths = [1]
        visited = set(queue)
        step = 0
        while step < len(queue):
            u = queue[step]
            depth = depths[step]
            step += 1
            for v in g.neighbors(u):
                if v not in visited:
                    queue.append(v)
                    depths.append(depth+1)
                    visited.add(v)
        is_connected = (len(queue) == g.number_of_nodes())
        return queue, depths, is_connected

    def collect_relations(self):
        g = self.graph
        nodes, depths, is_connected = self.bfs()
        relations = dict()
        for i, src in enumerate(nodes):
            relations[i] = dict()
            paths = nx.single_source_shortest_path(g, src)
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                assert tgt in paths
                path = paths[tgt]
                info = dict()
                info['edge'] = [ g[path[i]][path[i+1]]['label'] for i in range(len(path)-1) ]
                info['length'] = len(info['edge'])
                relations[i][j].append(info)
        items = [self.nid2iid[node] for node in nodes]
        return items, depths, relations, is_connected

def seq_to_relation(seq):
    seq_graph = SeqGraph(seq)
    items, depth, relation, ok  = seq_graph.collect_relations()
    assert ok, 'not connected'
    seq_select = [np.where(np.array(items) == seq[i])[0][0] for i in range(len(seq))]
    
    depth = [1] + depth
    items = [1] + items
    return items, depth, relation, seq_select


def padding_batch_lists(xs, pad=0):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [pad] * (max_len-len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data

def padding_batch_arrays(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([ list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
    return data


def collate_fn(samples):
    seqs, labels = zip(*samples)
    items = []
    depths = []
    relations = []
    seq_selects = []

    res = list(map(seq_to_relation, seqs))
    for item, depth, relation, seq_select in res:
        items.append(item)
        depths.append(depth)
        relations.append(relation)
        seq_selects.append(seq_select)

    all_relations = dict()
    pad_idx = relation_vocab.token2idx(PAD)
    cls_idx = relation_vocab.token2idx(CLS)
    rcls_idx = relation_vocab.token2idx(RCLS)
    self_idx = relation_vocab.token2idx(SEL)
    all_relations[tuple([pad_idx])] = 0
    all_relations[tuple([cls_idx])] = 1
    all_relations[tuple([rcls_idx])] = 2
    all_relations[tuple([self_idx])] = 3

    relation_type = []
    for id, relation in enumerate(relations):
        n = len(items[id])-1
        relation_matrix = [ [3]+[1]*(n) ]
        for i in range(n):
            rs = [2]
            for j in range(n):
                all_path = relation[i][j]
                path = random.choice(all_path)['edge']
                if len(path) == 0:
                    path = [SEL]
                if len(path) > 8:
                    path = [TL]
                path = tuple(relation_vocab.token2idx(path))
                rtype = all_relations.get(path, len(all_relations))
                if rtype == len(all_relations):
                    all_relations[path] = len(all_relations)
                rs.append(rtype)
            rs = np.array(rs, dtype=np.int)
            relation_matrix.append(rs)
        relation_matrix = np.stack(relation_matrix)
        relation_type.append(relation_matrix)
    relation_type = np.transpose(padding_batch_arrays(relation_type), (2, 1, 0))

    relation_bank = dict()
    relation_length = dict()
    for k, v in all_relations.items():
        relation_bank[v] = np.array(k, dtype=np.int)
        relation_length[v] = len(k)
    relation_bank = [relation_bank[i] for i in range(len(all_relations))]
    relation_length = [relation_length[i] for i in range(len(all_relations))]
    relation_bank = np.transpose(padding_batch_arrays(relation_bank))
    relation_length = np.array(relation_length)
    
    items = padding_batch_lists(items)
    seqs = padding_batch_lists(seqs)
    depths = padding_batch_lists(depths)
    labels = np.array(labels)
    seq_selects = padding_batch_lists(seq_selects)
    e = np.eye(items.shape[0]-1)
    seq_selects = e[seq_selects]

    data = {
        'items': items,
        'seqs': seqs,
        'depths': depths,
        'seq_selects': seq_selects,
        'relation_vocab': relation_vocab,
        'relations': relation_type,
        'relation_bank': relation_bank,
        'relation_length': relation_length,
        'labels': labels,
        'item_padding_idx': 0,
        'depth_padding_idx': 0,
    }
    return data
    
