import typing

import networkx as nx

import patterns
from templating import base, util


class StructuredLoopTemplate(base.BaseRuleTemplate):
    @staticmethod
    def pattern():
        pattern = nx.DiGraph()

        pattern.add_node("IncomingFlow", type="Flow")
        pattern.add_node("Merge", type="Exclusive")
        pattern.add_node("Ref1Flow", type="Flow")
        pattern.add_node("Ref2Flow", type="Flow")
        pattern.add_node("Split", type="Exclusive")
        pattern.add_node("RepeatFlow", type="Flow")
        pattern.add_node("OutgoingFlow", type="Flow")

        pattern.add_edge("IncomingFlow", "Merge")
        pattern.add_edge("Merge", "Ref1Flow")
        pattern.add_edge("Ref2Flow", "Split")
        pattern.add_edge("Split", "RepeatFlow")
        pattern.add_edge("RepeatFlow", "Merge")
        pattern.add_edge("Split", "OutgoingFlow")

        return pattern

    def generate(self, graph: nx.DiGraph) -> typing.List[base.UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        labels = nx.get_node_attributes(graph, "label")
        depths = patterns.get_node_depths(graph)

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            if depths[match["Split"]] < depths[match["Merge"]]:
                # not a real loop
                continue

            incoming_ref = patterns.get_predecessors_of_type(
                graph,
                match["IncomingFlow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            ref1 = patterns.get_successors_not_of_type(
                graph,
                match["Ref1Flow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            nodes = [
                match["IncomingFlow"],
                match["Ref1Flow"],
            ]
            content = [
                "It is obligatory that",
                base.ForwardReference(ref1),
                "after",
                base.ForwardReference(incoming_ref),
            ]

            rules.append(base.UnresolvedRule(
                content=content,
                nodes=nodes,
                depth=min(depths[n] for n in nodes)
            ))

            nodes = [
                match["Split"],
                match["RepeatFlow"],
                match["OutgoingFlow"],
            ]

            split_label = labels[match["Split"]]
            repeat_condition_label = labels[match["RepeatFlow"]]
            continue_condition_label = labels[match["OutgoingFlow"]]

            nodes.append(match["OutgoingFlow"])
            outgoing_ref = patterns.get_successors_not_of_type(
                graph,
                match["OutgoingFlow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            nodes.append(match["Ref1Flow"])
            ref2 = patterns.get_successors_not_of_type(
                graph,
                match["Ref2Flow"],
                types=["Actor", "Uses", "DataObject"]
            )[0]
            content = [
                "It is obligatory that either",
                base.ForwardReference(ref1),
                "is repeated"
            ]
            repeat_condition = split_label + " " + repeat_condition_label
            repeat_condition = repeat_condition.strip()
            if len(repeat_condition) > 0:
                content += [
                    "in case",
                    repeat_condition,
                ]

            content += [
                "or",
                base.ForwardReference(outgoing_ref),
            ]
            break_condition = split_label + " " + continue_condition_label
            break_condition = break_condition.strip()
            if len(break_condition) > 0:
                content += [
                    "in case",
                    break_condition,
                ]

            content += [
                "after",
                base.ForwardReference(ref2)
            ]
            rules.append(base.UnresolvedRule(
                content=content,
                nodes=nodes,
                depth=min(depths[n] for n in nodes)
            ))

        return rules
