# Attention: as shown on the table above
# nodes indexed from 1 to ...
# edges indexed from 0 to ...
#import networkx as nx
import numpy as np
import scipy.sparse as sp
import math

class TransportGraph:
    def __init__(self, graph_data, maxpath_const = 3):
        graph_table = graph_data['graph_table']
        
        self.kNodesNumber = graph_data['kNodesNumber']
        self.kLinksNumber = graph_data['kLinksNumber']
        self.kMaxPathLength = maxpath_const * int(math.sqrt(self.kNodesNumber))
        
        self.capacities_array = np.array(graph_table[['Capacity']]).flatten()
        self.free_flow_times_array = np.array(graph_table[['Free Flow Time']]).flatten()
        self.sources = np.empty(self.kLinksNumber, dtype='int32')
        self.targets = np.empty(self.kLinksNumber, dtype='int32')
        
        in_incident_matrix = sp.lil_matrix((self.kNodesNumber, self.kLinksNumber), dtype='int32')
        out_incident_matrix = sp.lil_matrix((self.kNodesNumber, self.kLinksNumber), dtype='int32')
        self.nodes_indices = {}
        index = 0
        for edge, row in enumerate(graph_table[['Init node', 'Term node']].itertuples()):
            if row[1] not in self.nodes_indices:
                self.nodes_indices[row[1]] = index
                index += 1
            source = self.nodes_indices[row[1]]
            self.sources[edge] = source
            out_incident_matrix[source, edge] = 1
            
            if row[2] not in self.nodes_indices:
                self.nodes_indices[row[2]] = index
                index += 1
            target = self.nodes_indices[row[2]]
            self.targets[edge] = target
            in_incident_matrix[target, edge] = 1
        
        in_incident_matrix = in_incident_matrix.tocsr()
        self.in_pointers = in_incident_matrix.indptr
        self.in_edges_array = in_incident_matrix.indices
        self.pred = self.sources[self.in_edges_array]
        
        out_incident_matrix = out_incident_matrix.tocsr()
        self.out_pointers = out_incident_matrix.indptr
        self.out_edges_array = out_incident_matrix.indices
        self.succ = self.targets[self.out_edges_array]
        
        """
        self.transport_graph = nx.DiGraph()
        
        self.transport_graph.add_nodes_from(np.arange(1, self.kNodesNumber + 1))
        
        for link_index in range(0, self.kLinksNumber):
            self.transport_graph.add_edge(self.graph_table.get_value(link_index, 0, takeable=True), 
                                          self.graph_table.get_value(link_index, 1, takeable=True),
                                          edge_index = link_index)
        """

    def capacities(self):
        #return np.array(self.graph_table[['Capacity']]).flatten()
        return np.array(self.capacities_array)
        
    def freeflowtimes(self):
        #return np.array(self.graph_table[['Free Flow Time']]).flatten()
        return np.array(self.free_flow_times_array)

    def nodes(self):
        #return self.transport_graph.nodes()
        return range(self.kNodesNumber)
      
    def edges(self):
        #return range(0, self.transport_graph.number_of_edges())
        return range(self.kLinksNumber)

    def successors(self, node_index):
        #return list(self.transport_graph.successors(vertex))
        return self.succ[self.out_pointers[node_index] : self.out_pointers[node_index + 1]]

    def predecessors(self, node_index):
        #return list(self.transport_graph.predecessors(vertex))
        return self.pred[self.in_pointers[node_index] : self.in_pointers[node_index + 1]]
    
    def get_nodes_indices(self, nodes):
        return np.array([self.nodes_indices[node] for node in nodes])
    
    def get_node_index(self, node):
        return self.nodes_indices[node]
        
    def in_edges(self, node_index):
        #return self._edges_indices(self.transport_graph.in_edges(vertex, data = True))
        return self.in_edges_array[self.in_pointers[node_index] : self.in_pointers[node_index + 1]]

    def out_edges(self, node_index):
        #return self._edges_indices(self.transport_graph.out_edges(vertex, data = True))
        return self.out_edges_array[self.out_pointers[node_index] : self.out_pointers[node_index + 1]]
      
    def source_of_edge(self, edge_index):
        #return self.graph_table.get_value(edge_index, 0, takeable=True)
        return self.sources[edge_index]
      
    def target_of_edge(self, edge_index):
        #return self.graph_table.get_value(edge_index, 1, takeable=True)
        return self.targets[edge_index]
