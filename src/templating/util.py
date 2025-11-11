import typing

import networkx as nx


def match_is_visited(graph: nx.Graph, match: typing.Dict[str, str]) -> bool:
    """
    Returns True, iff all nodes in the match are visited, False otherwise.
    :param graph: the process graph
    :param match: the matched pattern in the process graph
    :return:
    """
    for node in match.values():
        if not graph.nodes[node].get("visited", False):
            return False
    return True


def visit_match(graph: nx.Graph, match: typing.Dict[str, str]):
    visited = nx.get_node_attributes(graph, "visited")
    for node in match.values():
        visited[node] = True
    nx.set_node_attributes(graph, visited, "visited")


def visit_nodes(graph: nx.Graph, nodes: typing.List[str]):
    visited = nx.get_node_attributes(graph, "visited")
    for node in nodes:
        visited[node] = True
    nx.set_node_attributes(graph, visited, "visited")


def match_to_subgraph(graph: nx.Graph, match: typing.Dict[str, str]) -> nx.DiGraph:
    original_graph = graph
    graph = nx.DiGraph()
    graph.add_nodes_from(n for n in original_graph.nodes(data=True) if n in match.values())
    graph.add_edges_from(e for e in original_graph.edges(data=True) if e[0] in graph or e[1] in graph)
    return graph
