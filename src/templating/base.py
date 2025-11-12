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
    text: str


@dataclasses.dataclass
class UnresolvedRule:
    depth: int
    content: typing.List[typing.Union[str, "ForwardReference"]]
    nodes: typing.List[str]


class BaseRuleTemplate(abc.ABC):
    def __init__(self, include_tags: bool):
        self._include_tags = include_tags

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
    resolve_direction: typing.Literal["forward", "backward"]
