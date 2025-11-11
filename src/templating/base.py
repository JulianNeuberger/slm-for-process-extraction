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


class BaseRuleTemplate(abc.ABC):
    content: typing.List[str | "ForwardReference"]

    @abc.abstractmethod
    def generate(self, graph: nx.DiGraph) -> typing.List[Rule]:
        raise NotImplementedError()

    def to_text(self, rules_by_nodes: typing.Dict[str, Rule]) -> str:
        ret = []
        for c in self.content:
            if isinstance(c, str):
                ret += c
            elif isinstance(c, ForwardReference):
                ret += c.resolve(rules_by_nodes)
            else:
                raise TypeError(f"Unsupported rule content type {type(c)}.")
        return " ".join(ret)


class BaseFactTemplate(abc.ABC):
    @abc.abstractmethod
    def generate(self, graph: nx.DiGraph) -> typing.List[Fact]:
        raise NotImplementedError()


@dataclasses.dataclass
class ForwardReference:
    node: str

    def resolve(self, rules_by_nodes: typing.Dict[str, Rule]) -> str:
        return rules_by_nodes[self.node].ref_name