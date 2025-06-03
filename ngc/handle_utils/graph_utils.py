import os, pickle
import os.path as op
import numpy as np


class EasyGraph():
    """docstring for EasyGraph."""
    def __init__(self, num_nodes, edges, node_attrs=None, edge_attrs=None):
        self.num_nodes = num_nodes
        self.nodes_idx = np.arange(num_nodes)
        self.edges = edges
        # TODO: do check on node_attrs
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        if self.edge_attrs is None:
            self.edge_attrs = {}

        self.node_neighbor = [[] for i in range(num_nodes)]

        for e in edges:
            _i, _j = e
            self.node_neighbor[_i].append(_j)
            self.node_neighbor[_j].append(_i)

        self.degree = np.asarray([len(vnh) for vnh in self.node_neighbor ])
        if self.node_attrs is not None and 'name' in self.node_attrs:
            self.node_names = self.node_attrs['name']
        else:
            self.node_names = [f'node_{nid}' for nid in range(num_nodes)]

        self.name_dict = {name: num for num,name in enumerate(self.node_names)}


    def get_junc_and_path(self):
        # find all paths between endpoints
        paths = []
        junc_idx = self.nodes_idx[self.degree > 2]
        node_mark = np.zeros(self.num_nodes, dtype=bool)
        for node_id in junc_idx:
            node_ns = self.node_neighbor[node_id]
            for node_next in node_ns:
                if self.degree[node_next] != 2:
                    paths.append([node_id, node_next])
                    continue
                
                if not node_mark[node_next]:
                    chain = self.get_d2chain(node_id, node_next)
                    node_mark[chain[1:-1]] = True
                    paths.append(chain)
        
        return junc_idx, paths


    def get_d2chain(self, idx0, idx1):
        chain = [idx0, idx1]
        while (self.degree[chain[-1]] == 2):
            node1, node2 = self.node_neighbor[chain[-1]]
            next_vid = node1 if node2 == chain[-2] else node2

            # jump out for closed path
            if next_vid == idx0:
                break

            chain.append(next_vid)

        return chain
    
    def add_edge_attr(self, name, attrs):
        self.edge_attrs[name] = attrs
    

    def get_junc_neighbors(self, junc_name):
        node_idx = self.name_dict[junc_name]
        neighbors = self.node_neighbor[node_idx]
        curve_ids = [int(name.split('_')[1]) for name in self.node_names[neighbors]]
        return curve_ids
    

    def collect_ctrlpts(self):
        if 'control_t' not in self.edge_attrs:
            raise KeyError('No such attributes')
        
        edge_ctrl_pts = []
        for eid, e in enumerate(self.edges):
            # control_t must be in (0, 1)
            control_t = self.edge_attrs['control_t'][eid]

            i,j = e
            ctrl_info = {}
            name_i, name_j = self.node_names[i], self.node_names[j]
            if 'junc' in name_i and 'curve' in name_j:
                jid = int(name_i.split('_')[1])
                cid = int(name_j.split('_')[1])
                ctrl_info['side'] = 0
                ctrl_info['t'] = control_t
            elif 'junc' in name_j and 'curve' in name_i:
                jid = int(name_j.split('_')[1])
                cid = int(name_i.split('_')[1])
                ctrl_info['side'] = -1
                ctrl_info['t'] = 1 - control_t
            else:
                raise NameError('Wrong Collection?')
            
            ctrl_info['curve_id'] = cid
            ctrl_info['junc_id'] = jid
            edge_ctrl_pts.append([jid,cid, ctrl_info])

        return edge_ctrl_pts
    

    def print_info(self):
        print('#------------------------------------------#')
        print(f'Num of nodes: {self.num_nodes}, Num of edges: {len(self.edges)}')
        for e in self.edges:
            i,j = e
            if 'name' in self.node_attrs:
                name_i = self.node_attrs['name'][i]
                name_j = self.node_attrs['name'][j]
            else:
                name_i = f'node{i}'
                name_j = f'node{j}'

            print(f'Edge {name_i} -> {name_j}')
        print('#------------------------------------------#')


    def export_data(self):
        return {
            'num_nodes': self.num_nodes,
            'edges': self.edges,
            'node_attrs': self.node_attrs,
            'edge_attrs': self.edge_attrs
        }