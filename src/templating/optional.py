import typing

import networkx as nx

import patterns
from templating import base, util
from templating.base import UnresolvedRule


class OptionalRuleTemplate(base.BaseRuleTemplate):
    @staticmethod
    def pattern():
        pattern = nx.DiGraph()

        pattern.add_node("IncomingFlow", type="Flow")
        pattern.add_node("Split", type="Exclusive")
        pattern.add_node("Ref1Flow", type="Flow")
        pattern.add_node("SkipFlow", type="Flow")
        pattern.add_node("Merge", type="Exclusive")
        pattern.add_node("Ref2Flow", type="Flow")
        pattern.add_node("OutgoingFlow", type="Flow")

        pattern.add_edge("IncomingFlow", "Split")
        pattern.add_edge("Split", "Ref1Flow")
        pattern.add_edge("Split", "SkipFlow")
        pattern.add_edge("SkipFlow", "Merge")
        pattern.add_edge("Ref2Flow", "Merge")
        pattern.add_edge("Merge", "OutgoingFlow")

        return pattern

    def generate(self, graph: nx.DiGraph) -> typing.List[UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        labels = nx.get_node_attributes(graph, "label")
        depths = patterns.get_node_depths(graph)
        nodes = []

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            if depths[match["Split"]] > depths[match["Merge"]]:
                # not optional, probably loop
                continue

            skipped_ref = patterns.get_successors_of_type(
                graph,
                match["Ref1Flow"],
                ["Actor", "Uses", "DataObject"]
            )[0]

            nodes.append(match["Split"])
            nodes.append(match["SkipFlow"])
            skip_condition = labels[match["Split"]] + " " + labels[match["SkipFlow"]]
            skip_condition = skip_condition.strip()

            nodes.append(match["Ref1Flow"])
            execute_condition = labels[match["Split"]] + " " + labels[match["Ref1Flow"]]
            execute_condition = execute_condition.strip()

            content = ["It is permitted that optionally", base.ForwardReference(skipped_ref)]
            if len(execute_condition) > 0:
                content += [
                    "in case",
                    execute_condition,
                ]
            if len(skip_condition) > 0:
                content += [
                    "or not in case",
                    skip_condition,
                ]

            incoming_ref = patterns.get_predecessors_not_of_type(
                graph,
                match["IncomingFlow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            content += ["after", base.ForwardReference(incoming_ref)]

            depth = min([depths[n] for n in nodes])
            rules.append(base.UnresolvedRule(content=content, depth=depth, nodes=nodes))
            util.visit_nodes(graph, nodes)

            nodes = [match["Merge"], match["OutgoingFlow"]]
            outgoing_ref = patterns.get_successors_of_type(
                graph,
                match["OutgoingFlow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            nodes.append(match["Ref2Flow"])
            skipped_ref = patterns.get_predecessors_not_of_type(
                graph,
                match["Ref2Flow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            content = [
                "It is obligatory that",
                base.ForwardReference(outgoing_ref),
                "after one of",
                base.ForwardReference(incoming_ref),
                "or",
                base.ForwardReference(skipped_ref)
            ]
            depth = min(depths[n] for n in nodes)
            rules.append(base.UnresolvedRule(content=content, depth=depth, nodes=nodes))

        return rules
