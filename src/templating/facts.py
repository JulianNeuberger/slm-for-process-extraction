import typing

import networkx as nx

from templating import base


class TaskFactTemplate(base.BaseFactTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Fact]:
        facts = []
        for node, attr in graph.nodes(data=True):
            if attr["type"] == "Activity":
                facts.append(base.Fact(
                    text=attr["label"],
                    node=node,
                ))
        return facts


class ActorFactTemplate(base.BaseFactTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Fact]:
        facts = []
        for node, attr in graph.nodes(data=True):
            if attr["type"] == "Actor":
                facts.append(base.Fact(
                    text=attr["label"],
                    node=node,
                ))
        return facts