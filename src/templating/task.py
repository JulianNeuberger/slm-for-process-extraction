import typing

import networkx as nx

import patterns
from templating import base, util


class TaskRuleTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")

        depths = patterns.get_node_depths(graph)

        pattern = nx.Graph()
        pattern.add_node("Activity", type="Activity")
        pattern.add_node("Actor", type="Actor")
        pattern.add_edge("Activity", "Actor")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            activity_label = labels[match["Activity"]]
            actor_label = labels[match["Actor"]]

            text = f"{actor_label} {activity_label}"
            rules.append(base.Rule(
                text=text,
            ))

            util.visit_match(graph, match)
        return rules