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

    @staticmethod
    def generate_loop_body(graph: nx.DiGraph, match: typing.Dict[str, str], depths: typing.Dict[str, int]) -> base.UnresolvedRule:
        incoming_refs = patterns.get_predecessors_not_of_type(
            graph,
            match["IncomingFlow"],
            types=["Actor", "Uses", "DataObject"]
        )
        incoming_ref = incoming_refs[0]
        ref1 = patterns.get_successors_not_of_type(
            graph,
            match["Ref1Flow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]

        assert graph.nodes[ref1]["type"] != "Flow"
        assert graph.nodes[incoming_ref]["type"] != "Flow"
        content = [
            "It is obligatory that",
            base.ForwardReference(ref1, resolve_direction="forward"),
            "after",
            base.ForwardReference(incoming_ref, resolve_direction="backward"),
        ]

        nodes = [
            match["IncomingFlow"],
            match["Ref1Flow"],
            match["Merge"]
        ]
        util.visit_nodes(graph, nodes)
        return base.UnresolvedRule(
            content=content,
            nodes=nodes,
            depth=min(depths[n] for n in nodes)
        )

    @staticmethod
    def generate_loop_back(graph: nx.DiGraph, match: typing.Dict[str, str],
                           depths: typing.Dict[str, int]) -> base.UnresolvedRule:
        labels = nx.get_node_attributes(graph, "label")

        split_label = labels[match["Split"]]
        repeat_condition_label = labels[match["RepeatFlow"]]
        continue_condition_label = labels[match["OutgoingFlow"]]

        outgoing_ref = patterns.get_successors_not_of_type(
            graph,
            match["OutgoingFlow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]

        ref1 = patterns.get_successors_not_of_type(
            graph,
            match["Ref1Flow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]
        assert graph.nodes[ref1]["type"] != "Flow"
        content = [
            "It is obligatory that either",
            base.ForwardReference(ref1, resolve_direction="forward"),
            "is repeated"
        ]
        repeat_condition = split_label + " " + repeat_condition_label
        repeat_condition = repeat_condition.strip()
        if len(repeat_condition) > 0:
            content += [
                "in case",
                repeat_condition,
            ]

        assert graph.nodes[outgoing_ref]["type"] != "Flow"
        content += [
            "or",
            base.ForwardReference(outgoing_ref, resolve_direction="forward"),
        ]
        break_condition = split_label + " " + continue_condition_label
        break_condition = break_condition.strip()
        if len(break_condition) > 0:
            content += [
                "in case",
                break_condition,
            ]

        ref2 = patterns.get_predecessors_not_of_type(
            graph,
            match["Ref2Flow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]
        print("------", graph.nodes[ref2]["label"])
        assert graph.nodes[ref2]["type"] != "Flow"
        content += [
            "after",
            base.ForwardReference(ref2, resolve_direction="backward"),
        ]
        nodes = [
            match["Split"],
            match["RepeatFlow"],
            match["OutgoingFlow"],
            match["Ref2Flow"],
        ]
        util.visit_nodes(graph, nodes)
        return base.UnresolvedRule(
            content=content,
            nodes=nodes,
            depth=min(depths[n] for n in nodes)
        )

    def generate(self, graph: nx.DiGraph) -> typing.List[base.UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        depths = patterns.get_node_depths(graph)

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            if depths[match["Split"]] < depths[match["Merge"]]:
                # not a real loop
                continue

            rules.append(self.generate_loop_body(graph, match, depths))
            rules.append(self.generate_loop_back(graph, match, depths))

            util.assert_match_visited(graph, match)
        return rules
