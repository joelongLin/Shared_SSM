import numpy as np
from typing import NamedTuple,Optional
from networkx.algorithms.shortest_paths.generic import shortest_path
import networkx as nx

class Node(NamedTuple):
    name:str
    index:int
    father:int
    children_list:list

def turn_str_to_tree():

    pass

if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_edges_from([(1,2),(1,3),(2,4),(2,5),(5,8),(3,6)])
    result = shortest_path(graph , source=4 , target=6)
    print(result)

