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

    def generate_skip_rule(self,
                           graph: nx.DiGraph,
                           match: typing.Dict[str, str],
                           depths: typing.Dict[str, int]) -> base.UnresolvedRule:
        labels = nx.get_node_attributes(graph, "label")
        nodes = [match["Split"], match["SkipFlow"], match["Ref1Flow"], match["IncomingFlow"]]
        skipped_ref = patterns.get_successors_not_of_type(
            graph,
            match["Ref1Flow"],
            ["Actor", "Uses", "DataObject"]
        )[0]

        skip_condition = labels[match["Split"]] + " " + labels[match["SkipFlow"]]
        skip_condition = skip_condition.strip()

        execute_condition = labels[match["Split"]] + " " + labels[match["Ref1Flow"]]
        execute_condition = execute_condition.strip()

        assert graph.nodes[skipped_ref]["type"] != "Flow"
        content = [
            "It is obligatory that either",
            base.ForwardReference(skipped_ref, resolve_direction="forward")
        ]
        if len(execute_condition) > 0:
            content.append("in case")
            if self._include_tags:
                content.append("<cond>")
            content.append(execute_condition)
            if self._include_tags:
                content.append("</cond>")
        if len(skip_condition) > 0:
            content.append("or not in case")
            if self._include_tags:
                content.append("<cond>")
            content.append(skip_condition)
            if self._include_tags:
                content.append("</cond>")

        incoming_ref = patterns.get_predecessors_not_of_type(
            graph,
            match["IncomingFlow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]
        assert graph.nodes[incoming_ref]["type"] != "Flow"
        content += ["after", base.ForwardReference(incoming_ref, resolve_direction="backward")]

        depth = min([depths[n] for n in nodes])
        util.visit_nodes(graph, nodes)
        return base.UnresolvedRule(content=content, depth=depth, nodes=nodes)

    @staticmethod
    def generate_continue_rule(graph: nx.DiGraph, match: typing.Dict[str, str], depths: typing.Dict[str, int]) -> base.UnresolvedRule:
        incoming_ref = patterns.get_predecessors_not_of_type(
            graph,
            match["IncomingFlow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]
        assert graph.nodes[incoming_ref]["type"] != "Flow"
        nodes = [match["Merge"], match["OutgoingFlow"], match["Ref2Flow"]]
        outgoing_ref = patterns.get_successors_not_of_type(
            graph,
            match["OutgoingFlow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]
        skipped_ref = patterns.get_predecessors_not_of_type(
            graph,
            match["Ref2Flow"],
            types=["Actor", "Uses", "DataObject"]
        )[0]
        assert graph.nodes[outgoing_ref]["type"] != "Flow"
        assert graph.nodes[incoming_ref]["type"] != "Flow"
        assert graph.nodes[skipped_ref]["type"] != "Flow"
        content = [
            "It is obligatory that",
            base.ForwardReference(outgoing_ref, resolve_direction="forward"),
            "after one of",
            base.ForwardReference(incoming_ref, resolve_direction="backward"),
            "or",
            base.ForwardReference(skipped_ref, resolve_direction="backward"),
        ]
        depth = min(depths[n] for n in nodes)
        util.visit_nodes(graph, nodes)
        return base.UnresolvedRule(content=content, depth=depth, nodes=nodes)

    def generate(self, graph: nx.DiGraph) -> typing.List[UnresolvedRule]:
        rules = []
        matches = patterns.find_graph_pattern(graph, self.pattern())
        depths = patterns.get_node_depths(graph)

        for match in matches:
            if util.match_is_visited(graph, match):
                continue

            if depths[match["Split"]] > depths[match["Merge"]]:
                # not optional, probably loop
                continue

            rules.append(self.generate_skip_rule(graph, match, depths))
            rules.append(self.generate_continue_rule(graph, match, depths))
            util.assert_match_visited(graph, match)

        return rules
