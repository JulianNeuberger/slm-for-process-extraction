import abc
import typing

import networkx as nx

import patterns
from templating import base, util
from templating.base import ForwardReference


class BaseGatewayRuleTemplate(base.BaseRuleTemplate, abc.ABC):
    @abc.abstractmethod
    def gateway_type(self) -> str:
        pass

    @abc.abstractmethod
    def leading_clause(self):
        pass

    @abc.abstractmethod
    def join_clause(self):
        pass

    def pattern(self) -> nx.DiGraph:
        pattern = nx.DiGraph()
        pattern.add_node("FlowToGateway", type="Flow")
        pattern.add_node("Gateway", type=self.gateway_type())
        pattern.add_node("FlowOption1", type="Flow")
        pattern.add_node("FlowOption2", type="Flow")

        pattern.add_edge("FlowToGateway", "Gateway")
        pattern.add_edge("Gateway", "FlowOption1")
        pattern.add_edge("Gateway", "FlowOption2")

        return pattern

    def generate(self, graph: nx.DiGraph) -> typing.List[base.UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        labels = nx.get_node_attributes(graph, "label")
        depths = patterns.get_node_depths(graph)

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            options = []
            out_flows = patterns.get_successors_of_type(graph, match["Gateway"], ["Flow"])

            for out_flow in out_flows:
                options.append((
                    out_flow,
                    patterns.get_successors_not_of_type(
                        graph,
                        out_flow,
                        types=["DataObject", "Uses", "Actor"]
                    )[0]
                ))
            content = [
                "It is obligatory that",
                self.leading_clause(),
            ]
            gateway_label = labels[match["Gateway"]]
            first = True
            for flow, ref in options:
                if first:
                    first = False
                else:
                    content.append(self.join_clause())

                assert graph.nodes[ref]["type"] != "Flow"
                content.append(ForwardReference(ref, resolve_direction="forward"))

                flow_label = labels[flow]
                condition = f"{gateway_label} {flow_label}".strip()
                if len(condition) > 0:
                    content.append("if")
                    if self._include_tags:
                        content.append("<cond>")
                    content.append(condition)
                    if self._include_tags:
                        content.append("</cond>")

            incoming_ref = patterns.get_predecessors_not_of_type(
                graph,
                match["FlowToGateway"],
                types=["DataObject", "Uses", "Actor"]
            )[0]

            assert graph.nodes[incoming_ref]["type"] != "Flow"
            content += [
                f"after",
                base.ForwardReference(incoming_ref, resolve_direction="backward"),
            ]

            depth = min(depths[n] for n in match.values())
            nodes = [
                match["Gateway"],
                match["FlowToGateway"],
                *out_flows
            ]
            rules.append(base.UnresolvedRule(content=content, depth=depth, nodes=nodes))
            util.visit_nodes(graph, nodes)
            util.assert_match_visited(graph, match)

        return rules


class BaseGatewayMergeRuleTemplate(base.BaseRuleTemplate, abc.ABC):
    @abc.abstractmethod
    def gateway_type(self) -> str:
        pass

    @abc.abstractmethod
    def leading_clause(self):
        pass

    @abc.abstractmethod
    def join_clause(self):
        pass

    def pattern(self) -> nx.DiGraph:
        pattern = nx.DiGraph()
        pattern.add_node("FlowToGateway1", type="Flow")
        pattern.add_node("FlowToGateway2", type="Flow")
        pattern.add_node("Gateway", type=self.gateway_type())
        pattern.add_node("FlowFromGateway", type="Flow")

        pattern.add_edge("FlowToGateway1", "Gateway")
        pattern.add_edge("FlowToGateway2", "Gateway")
        pattern.add_edge("Gateway", "FlowFromGateway")

        return pattern

    def generate(self, graph: nx.DiGraph) -> typing.List[base.UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        depths = patterns.get_node_depths(graph)

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            content: typing.List[str | base.ForwardReference] = ["It is obligatory that"]

            outgoing_ref = patterns.get_successors_not_of_type(
                graph,
                match["FlowFromGateway"],
                types=["DataObject", "Uses", "Actor"]
            )[0]
            assert graph.nodes[outgoing_ref]["type"] != "Flow"
            content += [
                base.ForwardReference(outgoing_ref, resolve_direction="forward"),
                "after",
                self.leading_clause(),
            ]

            in_flows = patterns.get_predecessors_of_type(graph, match["Gateway"], ["Flow"])
            first = True
            for in_flow in in_flows:
                if first:
                    first = False
                else:
                    content.append(self.join_clause())

                predecessor_ref = patterns.get_predecessors_not_of_type(
                    graph,
                    in_flow,
                    types=["DataObject", "Uses", "Actor"]
                )[0]
                assert graph.nodes[predecessor_ref]["type"] != "Flow"
                content.append(ForwardReference(predecessor_ref, resolve_direction="backward"))

            depth = depths[match["Gateway"]]
            nodes = [
                match["Gateway"],
                match["FlowFromGateway"],
                *in_flows
            ]
            rules.append(base.UnresolvedRule(content=content, depth=depth, nodes=nodes))
            util.visit_nodes(graph, nodes)
            util.assert_match_visited(graph, match)

        return rules
