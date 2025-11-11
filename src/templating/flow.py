import typing

import networkx as nx

import mappings
import patterns
from templating import base, util


class SequenceFlowTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")
        types = nx.get_node_attributes(graph, "type")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("Left", type="Activity+StartEvent")
        pattern.add_node("Flow", type="Flow")
        pattern.add_node("Right", type="Activity+EndEvent")

        pattern.add_edge("Left", "Flow")
        pattern.add_edge("Flow", "Right")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue


            if types[match["Left"]] == "StartEvent":
                left_text = "the process starts"
            else:
                left_label = labels[match["Left"]]
                left_actor = patterns.get_actor(graph, match["Left"])
                left_actor_label = labels[left_actor]
                left_text = f"{left_actor_label} {left_label}"

            if types[match["Right"]] == "EndEvent":
                right_text = "the process ends"
            else:
                right_label = labels[match["Right"]]
                right_actor = patterns.get_actor(graph, match["Right"])
                right_actor_label = labels[right_actor]
                right_text = f"{right_actor_label} {right_label}"

            text = (f"It is obligatory that {right_text} "
                    f"after {left_text}")

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
        return rules
