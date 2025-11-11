import typing

import networkx as nx

import patterns
from templating import base, util


class ExclusiveChoiceTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")
        types = nx.get_node_attributes(graph, "type")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("Decision", type="Activity+StartEvent")
        pattern.add_node("FlowActivityExclusive", type="Flow")
        pattern.add_node("Exclusive", type="Exclusive")
        pattern.add_node("FlowExclusiveOption1", type="Flow")
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowExclusiveOption2", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")

        pattern.add_edge("Decision", "FlowActivityExclusive")
        pattern.add_edge("FlowActivityExclusive", "Exclusive")
        pattern.add_edge("Exclusive", "FlowExclusiveOption1")
        pattern.add_edge("FlowExclusiveOption1", "ActivityOption1")
        pattern.add_edge("Exclusive", "FlowExclusiveOption2")
        pattern.add_edge("FlowExclusiveOption2", "ActivityOption2")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            outcomes = patterns.get_successors_of_type(graph, match["Exclusive"], types=["Flow"])
            outcomes.sort(key=lambda x: len(labels[x]), reverse=True)

            outcome_texts = []
            mark_as_visited = []
            should_use_otherwise = len(outcomes) > 1 and all(len(o) > 0 for o in outcomes[:-1])
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
                if len(labels[outcome]) > 0:
                    outcome_text += f" in case {labels[outcome]}"
                elif i == len(outcomes) - 1 and should_use_otherwise:
                    outcome_text += " otherwise"
                outcome_texts.append(outcome_text)

            decision = match["Decision"]
            if types[decision] == "Activity":
                deciding_actor = patterns.get_actor(graph, match["Decision"])
                deciding_actor_label = labels[deciding_actor]
                decision_label = labels[match["Decision"]]
                decision_text = f"{deciding_actor_label} {decision_label}"
            else:
                decision_text = "the process starts"

            text = "It is obligatory that either "
            text += " or ".join(outcome_texts)
            text += f" but only one after {decision_text}"

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules


class ExplicitMergeTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")
        types = nx.get_node_attributes(graph, "type")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowOption1Merge", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")
        pattern.add_node("FlowOption2Merge", type="Flow")
        pattern.add_node("Merge", type="Exclusive")
        pattern.add_node("FlowExclusiveActivity", type="Flow")
        pattern.add_node("AfterMerge", type="Activity+EndEvent")

        pattern.add_edge("ActivityOption1", "FlowOption1Merge")
        pattern.add_edge("ActivityOption2", "FlowOption2Merge")
        pattern.add_edge("FlowOption1Merge", "Merge")
        pattern.add_edge("FlowOption2Merge", "Merge")
        pattern.add_edge("Merge", "FlowExclusiveActivity")
        pattern.add_edge("FlowExclusiveActivity", "AfterMerge")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            after_merge = match["AfterMerge"]
            if types[after_merge] == "Activity":
                activity_actor = patterns.get_actor(graph, after_merge)
                activity_actor_label = labels[activity_actor]
                activity_label = labels[after_merge]
                after_merge_text = f"{activity_actor_label} {activity_label}"
            else:
                after_merge_text = "the process ends"

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

            text = f"It is obligatory that {after_merge_text} after either "
            text += " or ".join(option_texts)

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules


class ImplicitMergeTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowOption1Merge", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")
        pattern.add_node("FlowOption2Merge", type="Flow")
        pattern.add_node("Merge", type="Activity")

        pattern.add_edge("ActivityOption1", "FlowOption1Merge")
        pattern.add_edge("ActivityOption2", "FlowOption2Merge")
        pattern.add_edge("FlowOption1Merge", "Merge")
        pattern.add_edge("FlowOption2Merge", "Merge")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            activity_actor = patterns.get_actor(graph, match["Merge"])
            activity_actor_label = labels[activity_actor]
            activity_label = labels[match["Merge"]]

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

            text = f"It is obligatory that {activity_actor_label} {activity_label} after either "
            text += " or ".join(option_texts)

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules
