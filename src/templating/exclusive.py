import typing

import networkx as nx

import patterns
from templating import gateway, base, util


class ExclusiveChoiceTemplate(gateway.BaseGatewayRuleTemplate):
    def gateway_type(self) -> str:
        return "Exclusive"

    def leading_clause(self):
        return "one of"

    def join_clause(self):
        return "or"


class ExplicitMergeTemplate(gateway.BaseGatewayMergeRuleTemplate):
    def gateway_type(self) -> str:
        return "Exclusive"

    def leading_clause(self):
        return "one of"

    def join_clause(self):
        return "or"


class ImplicitMergeTemplate(base.BaseRuleTemplate):
    def pattern(self) -> nx.DiGraph:
        pattern = nx.DiGraph()
        pattern.add_node("FlowToMerge1", type="Flow")
        pattern.add_node("FlowToMerge2", type="Flow")
        pattern.add_node("Merge", type="Activity+EndEvent")

        pattern.add_edge("FlowToMerge1", "Merge")
        pattern.add_edge("FlowToMerge2", "Merge")

        return pattern

    def generate(self, graph: nx.DiGraph) -> typing.List[base.UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        depths = patterns.get_node_depths(graph)

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            merge_ref = match["Merge"]
            content: typing.List[str | base.ForwardReference] = [
                "It is obligatory that",
                base.ForwardReference(merge_ref, resolve_direction="forward"),
                "after",
            ]

            in_flows = patterns.get_predecessors_of_type(graph, match["Merge"], ["Flow"])
            first = True
            for in_flow in in_flows:
                if first:
                    first = False
                else:
                    content.append("or")

                predecessor_ref = patterns.get_predecessors_not_of_type(
                    graph,
                    in_flow,
                    types=["DataObject", "Uses", "Actor"]
                )[0]
                assert graph.nodes[predecessor_ref]["type"] != "Flow"
                content.append(base.ForwardReference(predecessor_ref, resolve_direction="backward"))

            depth = min(depths[n] for n in match.values())
            nodes = [
                match["Merge"],
                *in_flows
            ]
            rules.append(base.UnresolvedRule(content=content, depth=depth, nodes=nodes))
            util.visit_nodes(graph, nodes)
            util.assert_match_visited(graph, match)

        return rules
    
    


if __name__ == "__main__":
    def main():
        pass

    main()