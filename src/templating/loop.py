import typing

import networkx as nx

import patterns
from templating import base, Rule, util


class StructuredLoopTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()

        pattern.add_node("Pre", type="Activity")
        pattern.add_node("PreToMergeFlow", type="Flow")
        pattern.add_node("Merge", type="Exclusive")
        pattern.add_node("MergeToRepeatedFlow", type="Flow")
        pattern.add_node("Repeated", type="Activity")
        pattern.add_node("RepeatedToSplitFlow", type="Flow")
        pattern.add_node("Split", type="Exclusive")
        pattern.add_node("SplitToMergeFlow", type="Flow")
        pattern.add_node("SplitToPostFlow", type="Flow")
        pattern.add_node("Post", type="Activity")

        pattern.add_edge("Pre", "PreToMergeFlow")
        pattern.add_edge("PreToMergeFlow", "Merge")
        pattern.add_edge("Merge", "MergeToRepeatedFlow")
        pattern.add_edge("MergeToRepeatedFlow", "Repeated")
        pattern.add_edge("Repeated", "RepeatedToSplitFlow")
        pattern.add_edge("RepeatedToSplitFlow", "Split")
        pattern.add_edge("Split", "SplitToMergeFlow")
        pattern.add_edge("Split", "SplitToPostFlow")
        pattern.add_edge("SplitToPostFlow", "Post")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            if depths[match["Split"]] < depths[match["Merge"]]:
                # not a real loop
                continue

            activity_actor = patterns.get_actor(graph, match["Pre"])
            activity_actor_label = labels[activity_actor]
            activity_label = labels[match["Pre"]]

            split_label = labels[match["Split"]]
            repeat_condition_label = labels[match["SplitToMergeFlow"]]
            continue_condition_label = labels[match["SplitToPostFlow"]]

            repeated_activity_label = labels[match["Repeated"]]
            repeated_activity_actor = patterns.get_actor(graph, match["Repeated"])
            repeated_activity_actor_label = labels[repeated_activity_actor]

            post_activity_label = labels[match["Post"]]
            post_activity_actor = patterns.get_actor(graph, match["Post"])
            post_activity_actor_label = labels[post_activity_actor]

            text = (f"It is permitted that "
                    f"{repeated_activity_actor_label} {repeated_activity_label} "
                    f"in case {split_label} {repeat_condition_label} "
                    f"or {post_activity_actor_label} {post_activity_label} "
                    f"in case {split_label} {continue_condition_label} "
                    f"after {activity_actor_label} {activity_label}")

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)

        return rules
