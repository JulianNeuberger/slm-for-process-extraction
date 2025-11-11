import typing

import networkx as nx

import patterns
from templating import base, util


class SequenceFlowTemplate(base.BaseRuleTemplate):
    @staticmethod
    def pattern():
        pattern = nx.DiGraph()
        pattern.add_node("Flow", type="Flow")

        return pattern

    def generate(self, graph: nx.DiGraph) -> typing.List[base.UnresolvedRule]:
        rules: typing.List[base.UnresolvedRule] = []
        depths = patterns.get_node_depths(graph)

        for match in patterns.find_graph_pattern(graph, self.pattern()):
            if util.match_is_visited(graph, match):
                continue

            left_ref = patterns.get_predecessors_not_of_type(
                graph,
                match["Flow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            assert graph.nodes[left_ref]["type"] != "Flow"
            right_ref = patterns.get_successors_not_of_type(
                graph,
                match["Flow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            assert graph.nodes[right_ref]["type"] != "Flow"

            content = [
                "It is obligatory that",
                base.ForwardReference(right_ref, resolve_direction="forward"),
                "after",
                base.ForwardReference(left_ref, resolve_direction="backward"),
            ]

            rules.append(base.UnresolvedRule(
                content=content,
                nodes=[match["Flow"]],
                depth=depths[match["Flow"]],
            ))

            util.visit_nodes(graph, [match["Flow"]])
            util.assert_match_visited(graph, match)
        return rules
