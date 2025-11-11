import abc
import dataclasses
import typing

import networkx as nx


@dataclasses.dataclass
class Fact:
    text: str
    node: str


@dataclasses.dataclass
class Rule:
    id: str
    ref_name: str
    depth_in_process: int
    text: str


@dataclasses.dataclass
class UnresolvedRule:
    depth: int
    content: typing.List[str | "ForwardReference"]
    nodes: typing.List[str]
    reference_anchor: typing.List[str | "ForwardReference"]


class BaseRuleTemplate(abc.ABC):
    @abc.abstractmethod
    def generate(self, graph: nx.DiGraph) -> typing.List[UnresolvedRule]:
        raise NotImplementedError()


class BaseFactTemplate(abc.ABC):
    @abc.abstractmethod
    def generate(self, graph: nx.DiGraph) -> typing.List[Fact]:
        raise NotImplementedError()


@dataclasses.dataclass
class ForwardReference:
    node: str

    def resolve(self, rules_by_nodes: typing.Dict[str, Rule]) -> str:
        return rules_by_nodes[self.node].ref_name