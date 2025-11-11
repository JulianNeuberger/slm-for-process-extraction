import typing

import networkx as nx

import patterns
from templating import base, util


class InclusiveSplitTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("ActivityDecision", type="Activity")
        pattern.add_node("FlowActivityInclusive", type="Flow")
        pattern.add_node("Inclusive", type="Inclusive")
        pattern.add_node("FlowInclusiveOption1", type="Flow")
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowInclusiveOption2", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")

        pattern.add_edge("ActivityDecision", "FlowActivityInclusive")
        pattern.add_edge("FlowActivityInclusive", "Inclusive")
        pattern.add_edge("Inclusive", "FlowInclusiveOption1")
        pattern.add_edge("FlowInclusiveOption1", "ActivityOption1")
        pattern.add_edge("Inclusive", "FlowInclusiveOption2")
        pattern.add_edge("FlowInclusiveOption2", "ActivityOption2")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            deciding_actor = patterns.get_actor(graph, match["ActivityDecision"])
            deciding_actor_label = labels[deciding_actor]
            decision_label = labels[match["ActivityDecision"]]

            outcomes = patterns.get_successors_of_type(graph, match["Inclusive"], types=["Flow"])
            outcomes.sort(key=lambda x: len(labels[x]), reverse=True)

            outcome_texts = []
            mark_as_visited = []
            for i, outcome in enumerate(outcomes):
                options = patterns.get_successors_of_type(graph, outcome, types=["Activity"])
                if len(options) == 0:
                    continue
                assert len(options) == 1
                option = options[0]

                mark_as_visited.append(option)
                mark_as_visited.append(outcome)

                option_actor = labels[patterns.get_actor(graph, option)]
                outcome_text = f"{option_actor} {labels[option]}"
                outcome_texts.append(outcome_text)

            text = "It is obligatory that at least one of "
            text += " or ".join(outcome_texts)
            text += f" after {deciding_actor_label} {decision_label}"

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules
    

class StructuredSynchronizingMergeTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowOption1Merge", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")
        pattern.add_node("FlowOption2Merge", type="Flow")
        pattern.add_node("Merge", type="Inclusive")
        pattern.add_node("FlowInclusiveActivity", type="Flow")
        pattern.add_node("Activity", type="Activity")

        pattern.add_edge("ActivityOption1", "FlowOption1Merge")
        pattern.add_edge("ActivityOption2", "FlowOption2Merge")
        pattern.add_edge("FlowOption1Merge", "Merge")
        pattern.add_edge("FlowOption2Merge", "Merge")
        pattern.add_edge("Merge", "FlowInclusiveActivity")
        pattern.add_edge("FlowInclusiveActivity", "Activity")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            activity_actor = patterns.get_actor(graph, match["Activity"])
            activity_actor_label = labels[activity_actor]
            activity_label = labels[match["Activity"]]

            incoming_flows = patterns.get_predecessors_of_type(graph, match["Merge"], types=["Flow"])
            option_texts = []
            mark_as_visited = []
            for flow in incoming_flows:
                options = patterns.get_predecessors_of_type(graph, flow, types=["Activity"])
                if len(options) == 0:
                    continue
                assert len(options) == 1
                option = options[0]

                mark_as_visited.append(option)
                mark_as_visited.append(flow)

                option_actor = labels[patterns.get_actor(graph, option)]
                option_texts.append(f"{option_actor} {labels[option]}")

            text = f"It is obligatory that {activity_actor_label} {activity_label} after at least one of "
            text += " or ".join(option_texts)

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules
