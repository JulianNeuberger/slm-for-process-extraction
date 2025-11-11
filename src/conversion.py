import csv
import dataclasses
import json
import typing

import matplotlib.pyplot as plt
import networkx as nx


@dataclasses.dataclass
class NodeInfo:
    id: str
    label: typing.Optional[str]
    type: str


class Visitor:
    def __init__(self, stencil_mapping: typing.Optional[typing.Dict[str, str]]):
        self.nodes: typing.Dict[str, NodeInfo] = {}
        self.edges: typing.List[typing.Tuple[str, str]] = []

        self._stencil_mapping = stencil_mapping

    def __call__(self, shape: typing.Dict, parents: typing.List[typing.Dict]) -> None:
        stencil = shape["stencil"]["id"]
        name = shape["properties"].get("name", "").strip()
        shape_id: str = shape["resourceId"]

        if stencil not in self._stencil_mapping:
            return

        if stencil == "Pool" and len(name) == 0:
            # skip empty pools
            return

        self.nodes[shape_id] = NodeInfo(id=shape_id,
                                        label=name,
                                        type=self._stencil_mapping[stencil])

        for target in shape.get("outgoing", []):
            edge = (shape_id, target["resourceId"])
            self.edges.append(edge)

        # make connection to executing actor
        executing_actor: typing.Optional[typing.Dict] = None
        actor_candidates = []
        for parent in reversed(parents):
            if parent["stencil"]["id"] in ["Lane", "Pool"]:
                actor_candidates.append(parent)
                parent_name = parent["properties"].get("name", "").strip()
                if len(parent_name) != 0:
                    executing_actor = parent
                    break
        if executing_actor is None and len(actor_candidates) > 0:
            executing_actor = actor_candidates[0]
        if executing_actor is not None:
            executor_id: str = executing_actor["resourceId"]
            self.edges.append((shape_id, executor_id))


def traverse(shape: typing.Dict,
             parents: typing.List[typing.Dict],
             visitor: typing.Callable[[typing.Dict, typing.List[typing.Dict]], None]) -> None:
    visitor(shape, parents)
    for child in shape["childShapes"]:
        traverse(child, parents + [shape], visitor)


def sam_json_to_networkx(sam_json: typing.Dict, stencil_mapping: typing.Dict[str, str]) -> nx.DiGraph:
    visitor = Visitor(stencil_mapping)
    traverse(sam_json, [], visitor)

    g = nx.DiGraph()
    g.add_nodes_from((k, {"label": v.label, "type": v.type}) for k, v in visitor.nodes.items())
    g.add_edges_from((s, t) for s, t in visitor.edges if s in visitor.nodes and t in visitor.nodes)
    return g


def draw_process_graph(g: nx.DiGraph):
    pos = nx.spring_layout(g)
    labels = {
        n: f"{l} ({t})"
        for (n, t), (_, l)
        in zip(
            nx.get_node_attributes(g, "type").items(),
            nx.get_node_attributes(g, "label").items()
        )
    }

    colors = nx.get_node_attributes(g, "color", "blue")

    num_nodes = len(g.nodes)
    num_attrs = len(nx.get_node_attributes(g, "type"))
    assert num_nodes == num_attrs, f"{num_nodes} != {num_attrs}"
    nx.draw_networkx_nodes(g, pos, node_color=list(colors.values()))
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
