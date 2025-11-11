import collections
import typing

import networkx as nx
from networkx.algorithms import isomorphism

import mappings


def nodes_of_types(graph: nx.Graph, types: typing.Iterable[str]) -> typing.Generator[str, None, None]:
    node_types = nx.get_node_attributes(graph, "type")
    for node, t in node_types.items():
        if t in types:
            yield node


def find_edge_patterns(graph: nx.DiGraph,
                       *,
                       source_type: str,
                       target_type: str,
                       undirected: bool = False) -> typing.Generator[typing.Dict[str, typing.List[str]], None, None]:
    expected_types = {source_type, target_type}
    seen_edges: typing.Set[typing.Tuple[str, str]] = set()
    node_types = nx.get_node_attributes(graph, "type")
    for source in nodes_of_types(graph, [source_type]):
        edges = list(graph.out_edges(source))
        if undirected:
            edges += list(graph.in_edges(source))
        for edge in edges:
            if edge in seen_edges:
                continue
            target = [n for n in edge if n != source][0]
            actual_types = {node_types[source], node_types[target]}
            if expected_types == actual_types:
                ret = {}
                if source_type not in ret:
                    ret[source_type] = []
                if target_type not in ret:
                    ret[target_type] = []
                ret[source_type].append(source)
                ret[target_type].append(target)
                yield ret


# def find_pattern(graph: nx.DiGraph,
#                  *,
#                  source_type: str,
#                  target_types: typing.Iterable[str],
#                  undirected: bool = False) -> typing.Generator[typing.Dict[str, typing.List[str]], None, None]:
#     node_types = nx.get_node_attributes(graph, "type")
#     for source in nodes_of_types(graph, [source_type]):
#         edges = list(graph.out_edges(source))
#         if undirected:
#             edges += list(graph.in_edges(source))
#         missing_target_types = list(target_types)
#         relevant_edges = []
#         for edge in edges:
#             target = [n for n in edge if n != source][0]
#             target_type = node_types[target]
#             if target_type in missing_target_types:
#                 missing_target_types.remove(target_type)
#                 relevant_edges.append(edge)
#             if len(missing_target_types) == 0:
#                 # TODO: build combinations...
#                 break
#         if len(missing_target_types) == 0:
#             ret = collections.defaultdict(list)
#             ret[source_type].append(source)
#             for e in relevant_edges:
#                 target = [n for n in e if n != source][0]
#                 target_type = node_types[target]
#                 ret[target_type].append(target)
#             yield dict(**ret)


def get_actor(graph: nx.DiGraph, node: str, actor_type="Actor") -> typing.Optional[str]:
    for n in neighbors(graph, node, direction="ignore"):
        if graph.nodes[n]["type"] == actor_type:
            return n
    # print(f"{graph.nodes[node]['type']} '{graph.nodes[node]['label']}' not connected to any actor.")
    # print("connected to :")
    # for n in neighbors(graph, node, direction="ignore"):
    #     print(f"\t{graph.nodes[n]['label']} ({graph.nodes[n]['type']})")
    return None


def neighbors(graph: nx.DiGraph,
              node: str,
              direction: typing.Literal["forward", "reverse", "ignore"]) -> typing.Iterable[str]:
    def _neighbor_from_edge(edge: typing.Tuple[str, str]) -> str:
        if edge[0] != node:
            return edge[0]
        return edge[1]

    if direction == "forward":
        return list(_neighbor_from_edge(e) for e in graph.out_edges(node))
    elif direction == "reverse":
        return list(_neighbor_from_edge(e) for e in graph.in_edges(node))
    else:
        return (list(_neighbor_from_edge(e) for e in graph.in_edges(node)) +
                list(_neighbor_from_edge(e) for e in graph.out_edges(node)))


def get_node_depths(graph: nx.DiGraph, behaviour_element_types: typing.List[str] = None) -> typing.Dict[str, int]:
    if behaviour_element_types is None:
        behaviour_element_types = mappings.SapSamMappingCollection().behaviour.values()
    original_graph = graph
    graph = nx.DiGraph()
    graph.add_nodes_from(n for n in original_graph.nodes
                         if original_graph.nodes[n]["type"]
                         in behaviour_element_types)
    graph.add_edges_from(e for e in original_graph.edges
                         if e[0] in graph.nodes
                         and e[1] in graph.nodes)
    assert nx.is_connected(graph.to_undirected())
    root_candidates = [n for n, degree in graph.in_degree if degree == 0]
    root_candidates.sort(key=lambda n: len(original_graph.nodes[n]['label']), reverse=True)
    if len(root_candidates) > 1:
        print(f"Multiple roots: {' '.join(n + ' (' + original_graph.nodes[n]['label'] + ', ' + original_graph.nodes[n]['type'] + ')' for n in root_candidates)}")
    shortest_paths = {}
    for root in root_candidates:
        for n, depth in nx.shortest_path_length(graph, source=root).items():
            if n not in shortest_paths:
                shortest_paths[n] = depth
            else:
                shortest_paths[n] = min(shortest_paths[n], depth)
    return shortest_paths


def find_graph_pattern(graph: nx.DiGraph,
                       pattern: typing.Union[nx.Graph, nx.DiGraph]) -> typing.Generator[
    typing.Dict[str, str], None, None]:
    """
    Returns generator of dictionaries of isomorphic matches of "pattern" and subgraphs of "graph".
    Each entry in the generator is a mapping of nodes in "pattern" to nodes in "graph".

    Example:

    ```
    G = nx.DiGraph()
    G.add_node("a", type="Activity")
    G.add_node("b", type="Activity")
    G.add_node("c", type="Role")
    G.add_edge("a", "b")
    G.add_edge("a", "c")

    H = nx.DiGraph()
    H.add_node("Activity", type="Activity")
    H.add_node("Role", type="Role")
    H.add_edge("Role", "Activity")

    next(find_graph_pattern(G, H))
    ```

    returns

    ```
    {"Activity": "a", "Role": "c"}
    ```

    Attribute "type" of each node may contain more than one type, e.g., "Activity+StartEvent" will
    match nodes that are either Activity or StartEvent.

    :param graph: Graph, for which subgraphs will be checked for isomorphism with "pattern".
    :param pattern: Pattern to find in "graph".
    :return: generator of dictionaries of isomorphic matches of "pattern" to "graph".
    """

    def _node_match(n1, n2):
        return n1["type"] in n2["type"].split("+")

    if isinstance(pattern, nx.DiGraph):
        matcher = isomorphism.DiGraphMatcher(graph, pattern, _node_match)
    elif isinstance(pattern, nx.Graph):
        graph = graph.to_undirected()
        matcher = isomorphism.GraphMatcher(graph, pattern, _node_match)
    else:
        raise TypeError("Pattern must be either a DiGraph or a Graph")

    for match in matcher.subgraph_isomorphisms_iter():
        yield {v: k for k, v in match.items()}


def get_successors_of_type(graph: nx.DiGraph, node: str, types: typing.List[str]) -> typing.List[str]:
    successors = []
    for edge in graph.out_edges(node):
        if graph.nodes[edge[1]]["type"] not in types:
            continue
        successors.append(edge[1])
    return successors


def get_successors_not_of_type(graph: nx.DiGraph, node: str, types: typing.List[str]) -> typing.List[str]:
    successors = []
    for edge in graph.out_edges(node):
        if graph.nodes[edge[1]]["type"] in types:
            continue
        successors.append(edge[1])
    return successors


def get_predecessors_of_type(graph: nx.DiGraph, node: str, types: typing.List[str]) -> typing.List[str]:
    predecessors = []
    for edge in graph.in_edges(node):
        if graph.nodes[edge[0]]["type"] not in types:
            continue
        predecessors.append(edge[0])
    return predecessors

def get_predecessors_not_of_type(graph: nx.DiGraph, node: str, types: typing.List[str]) -> typing.List[str]:
    predecessors = []
    for edge in graph.in_edges(node):
        if graph.nodes[edge[0]]["type"] in types:
            continue
        predecessors.append(edge[0])
    return predecessors
