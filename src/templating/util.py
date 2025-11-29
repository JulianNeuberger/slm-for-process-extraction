import typing

import networkx as nx

import patterns
from templating import base


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


def resolve_reference(ref: base.ForwardReference,
                      rule_id_by_node: typing.Dict[str, int],
                      graph: nx.DiGraph,
                      with_tags: bool) -> typing.List[str | base.ForwardReference]:
    node_type = graph.nodes[ref.node]["type"]
    if node_type == "StartEvent":
        nx.set_node_attributes(graph, {ref.node: True}, "visited")
        return ["the process starts"]
    if node_type == "EndEvent":
        nx.set_node_attributes(graph, {ref.node: True}, "visited")
        return ["the process ends"]
    if node_type == "Activity":
        nx.set_node_attributes(graph, {ref.node: True}, "visited")
        actor = patterns.get_actor(graph, ref.node)
        actor_label = graph.nodes[actor]["label"]
        node_label = graph.nodes[ref.node]["label"]
        if with_tags:
            return [f"<actor> {actor_label} </actor> <activity> {node_label} </activity>"]
        return [f"{actor_label} {node_label}"]
    if node_type in ["Exclusive", "Inclusive", "Parallel"]:
        return [f"R{rule_id_by_node[ref.node]}"]

        join_clause = "and"
        if node_type == "Exclusive" or node_type == "Inclusive":
            join_clause = "or"

        if ref.resolve_direction == "forward":
            flows = patterns.get_successors_of_type(graph, ref.node, types=["Flow"])
            refs = [
                patterns.get_successors_not_of_type(
                    graph,
                    f,
                    types=["Uses", "DataObject", "Actor"]
                )[0] for f in flows
            ]
        else:
            flows = patterns.get_predecessors_of_type(graph, ref.node, types=["Flow"])
            refs = [
                patterns.get_predecessors_not_of_type(
                    graph,
                    f,
                    types=["Uses", "DataObject", "Actor"]
                )[0] for f in flows
            ]

        ret = []
        first = True
        for r in refs:
            if first:
                first = False
            else:
                ret.append(join_clause)
            assert graph.nodes[r]["type"] != "Flow"
            ret.append(base.ForwardReference(r, resolve_direction=ref.resolve_direction))
        return ret
    raise TypeError(f"{ref.node} has unsupported type {node_type}")


def assert_match_visited(graph: nx.DiGraph, match: typing.Dict[str, str]):
    assert match_is_visited(graph, match), [f'{graph.nodes[n]["type"]}: {graph.nodes[n].get("visited", False)}'
                                            for n in match.values()]
