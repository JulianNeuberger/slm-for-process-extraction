import typing

import networkx as nx

import patterns
from templating import base, util


class ParallelSplitTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")
        types = nx.get_node_attributes(graph, "type")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("BeforeSplit", type="Activity+StartEvent")
        pattern.add_node("FlowBeforeSplit", type="Flow")
        pattern.add_node("Split", type="Parallel")
        pattern.add_node("FlowSplitOption1", type="Flow")
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowSplitOption2", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")

        pattern.add_edge("BeforeSplit", "FlowBeforeSplit")
        pattern.add_edge("FlowBeforeSplit", "Split")
        pattern.add_edge("Split", "FlowSplitOption1")
        pattern.add_edge("FlowSplitOption1", "ActivityOption1")
        pattern.add_edge("Split", "FlowSplitOption2")
        pattern.add_edge("FlowSplitOption2", "ActivityOption2")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue
            
            before_split = match["BeforeSplit"]
            if types[before_split] == "Activity":
                split_actor = patterns.get_actor(graph, match["BeforeSplit"])
                split_actor_label = labels[split_actor]
                split_label = labels[match["BeforeSplit"]]
                before_split_text = f"{split_actor_label} {split_label}"
            else:
                before_split_text = "the process starts"

            outgoing_flows = patterns.get_successors_of_type(graph, match["Split"], types=["Flow"])
            mark_as_visited = []
            for flow in outgoing_flows:
                activities = patterns.get_successors_of_type(graph, flow, types=["Activity"])
                if len(activities) == 0:
                    continue
                assert len(activities) == 1
                activity = activities[0]

                mark_as_visited.append(activity)
                mark_as_visited.append(flow)

                activity_actor = labels[patterns.get_actor(graph, activity)]
                text = (f"It is obligatory that {activity_actor} {labels[activity]} "
                        f"after {before_split_text}")
                rules.append(base.Rule(
                    text=text,
                    depth_in_process=min(depths[n] for n in match.values() if n in depths),
                    described_sub_graph=util.match_to_subgraph(graph, match),
                ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules


class SynchronizationTemplate(base.BaseRuleTemplate):
    def generate(self, graph: nx.DiGraph) -> typing.List[base.Rule]:
        rules = []
        labels = nx.get_node_attributes(graph, "label")
        types = nx.get_node_attributes(graph, "type")

        depths = patterns.get_node_depths(graph)

        pattern = nx.DiGraph()
        pattern.add_node("ActivityOption1", type="Activity")
        pattern.add_node("FlowOption1Synchronization", type="Flow")
        pattern.add_node("ActivityOption2", type="Activity")
        pattern.add_node("FlowOption2Synchronization", type="Flow")
        pattern.add_node("Synchronization", type="Parallel")
        pattern.add_node("FlowSynchronizationActivity", type="Flow")
        pattern.add_node("AfterSync", type="Activity+EndEvent")

        pattern.add_edge("ActivityOption1", "FlowOption1Synchronization")
        pattern.add_edge("FlowOption1Synchronization", "Synchronization")
        pattern.add_edge("ActivityOption2", "FlowOption2Synchronization")
        pattern.add_edge("FlowOption2Synchronization", "Synchronization")
        pattern.add_edge("Synchronization", "FlowSynchronizationActivity")
        pattern.add_edge("FlowSynchronizationActivity", "AfterSync")

        for match in patterns.find_graph_pattern(graph, pattern):
            if util.match_is_visited(graph, match):
                continue

            activity_texts = []
            mark_as_visited = []
            incoming_flows = patterns.get_predecessors_of_type(graph, match["Synchronization"], types=["Parallel"])
            for flow in incoming_flows:
                activities = patterns.get_predecessors_of_type(graph, flow, types=["Activity"])
                if len(activities) == 0:
                    continue
                assert len(activities) == 1

                activity = activities[0]
                activity_actor = patterns.get_actor(graph, activity)
                activity_texts.append(f"{activity_actor} {labels[activity]}")

                mark_as_visited.append(activity)
                mark_as_visited.append(flow)

            after_sync = match["AfterSync"]
            if types[after_sync] == "Activity":
                synchronization_actor = patterns.get_actor(graph, after_sync)
                synchronization_actor_label = labels[synchronization_actor]
                activity_label = labels[after_sync]
                after_sync_text = f"{synchronization_actor_label} {activity_label}"
            else:
                after_sync_text = "the process ends"

            text = (f"It is obligatory that {after_sync_text} "
                    f"after all {' and '.join(activity_texts)}")

            rules.append(base.Rule(
                text=text,
                depth_in_process=min(depths[n] for n in match.values() if n in depths),
                described_sub_graph=util.match_to_subgraph(graph, match),
            ))

            util.visit_match(graph, match)
            util.visit_nodes(graph, mark_as_visited)

        return rules
