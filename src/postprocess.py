import abc
import csv
import json
import typing
import uuid

import matplotlib.pyplot as plt
import networkx as nx
import spacy
from nltk.corpus import wordnet
from spacy.tokens import Token

import conversion
import mappings
import patterns


class BaseProcessor(abc.ABC):
    @abc.abstractmethod
    def process(self, graph: nx.DiGraph):
        raise NotImplementedError()


class NounActivityProcessor(BaseProcessor):
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def get_verb_for_noun_activity(label: str) -> typing.Optional[str]:
        lemmas = wordnet.lemmas(label)
        if len(lemmas) == 0:
            # can't resolve, skip this activity
            return None
        for lemma in lemmas:
            for related_form in lemma.derivationally_related_forms():
                if related_form.synset().pos() == "v":
                    return related_form.name()
        return None

    def process_activity(self, graph: nx.DiGraph, node: str, node_labels: typing.Dict[str, str]):
        label = node_labels[node]
        if label.strip() == "":
            return
        label_doc = self._nlp(label)

        root: typing.Optional[Token] = None
        for token in label_doc:
            if token.dep_ == "ROOT":
                root = token
                if token.pos_ == "VERB":
                    return

        assert root is not None, ", ".join(f"{t.text} ({t.dep_})" for t in label_doc) + f" from: '{label}' ({graph.nodes[node]['type']})"
        # no predicate in this activity label, is it a Noun?
        if root.pos_ == "NOUN":
            verb_form = self.get_verb_for_noun_activity(label)
            if verb_form is not None:
                node_labels[node] = verb_form
                nx.set_node_attributes(graph, node_labels, "label")

    def process(self, graph: nx.DiGraph):
        node_types = nx.get_node_attributes(graph, "type")
        node_labels = nx.get_node_attributes(graph, "label")
        for node, node_type in node_types.items():
            if node_type not in ["Activity"]:
                continue
            self.process_activity(graph, node, node_labels)


class DataObjectAssociationProcessor(BaseProcessor):

    def process(self, graph: nx.DiGraph):
        node_labels = nx.get_node_attributes(graph, "label")
        node_types = nx.get_node_attributes(graph, "type")
        for match in patterns.find_edge_patterns(graph, source_type="Uses", target_type="Flow", undirected=True):
            association = match["Uses"][0]
            association_edges = list(graph.in_edges(association)) + list(graph.out_edges(association))
            data_object = None
            for l, r in association_edges:
                if node_types[l] == "DataObject":
                    data_object = l
                    break
                if node_types[r] == "DataObject":
                    data_object = r
                    break
            if data_object is None:
                print(f"WARN: Skipping Uses "
                      f"'{association}' connected to a Flow, "
                      f"as it has no connected DataObject.")
                continue

            flow = match["Flow"][0]
            if graph.has_edge(association, flow):
                graph.remove_edge(association, flow)
            else:
                graph.remove_edge(flow, association)

            flow_edges = list(graph.in_edges(flow)) + list(graph.out_edges(flow))
            assert len(flow_edges) == 2
            activities = [n for e in flow_edges for n in e if n != flow]
            assert len(activities) == 2

            data_object_label = node_labels[data_object]
            data_object_copy = data_object + "_copy"
            graph.add_node(data_object_copy, type="DataObject", label=data_object_label)
            association_copy = association + "_copy"
            graph.add_node(association_copy, type="Uses", label="")
            graph.add_edge(association_copy, data_object_copy)

            graph.add_edge(association_copy, activities[0])
            graph.add_edge(association, activities[1])


class MultiLineLabelProcessor(BaseProcessor):
    def process(self, graph: nx.DiGraph):
        node_labels = nx.get_node_attributes(graph, "label")
        for node in graph.nodes:
            node_labels[node] = node_labels[node].replace("\n", " ")
        nx.set_node_attributes(graph, node_labels, "label")


class DataObjectLabelProcessor(BaseProcessor):
    def __init__(self):
        self._nlp = spacy.load("en_core_web_trf")

    def data_object_from_activity(self, activity_label: str) -> typing.Optional[str]:
        activity_doc = self._nlp(activity_label)
        for token in activity_doc:
            if token.dep_ == "dobj" and token.pos_ == "NOUN":
                subtokens = sorted(token.subtree, key=lambda x: x.idx)
                return "".join(t.text_with_ws for t in subtokens)
        return None

    def supplement_activity_with_data_object(self, activity_label: str, data_object_label: str) -> typing.Optional[str]:
        pass

    def activity_label_well_formed(self, activity_label: str) -> bool:
        activity_doc = self._nlp(activity_label)
        has_predicate = False
        has_object = False
        for t in activity_doc:
            if t.dep_ == "ROOT" and t.pos_ == "VERB":
                has_predicate = True
            if t.dep_ == "dobj" and t.pos_ == "NOUN":
                has_object = True
        return has_predicate and has_object

    def data_object_well_formed(self, data_object_label) -> bool:
        data_object_doc = self._nlp(data_object_label)
        data_object_root = next(data_object_doc.sents).root
        if data_object_root.pos_ == "NOUN":
            # assume this is a well-formed data object
            return True
        return False

    def build_labels(self, data_object_label: str, activity_label: str) -> typing.Tuple[str, str]:
        activity_doc = self._nlp(activity_label)

        activity_object: typing.Optional[Token] = None
        for token in activity_doc:
            if token.dep_ == "dobj":
                activity_object = token

        if data_object_label.lower() in activity_label.lower():
            return activity_label, data_object_label

    def process(self, graph: nx.DiGraph):
        node_labels = nx.get_node_attributes(graph, "label")

        pattern = nx.Graph()
        pattern.add_node("Activity", type="Activity")
        pattern.add_node("DataObject", type="DataObject")
        pattern.add_node("Uses", type="Uses")
        pattern.add_edge("Activity", "Uses")
        pattern.add_edge("DataObject", "Uses")

        for match in patterns.find_graph_pattern(graph, pattern):
            association = match["Uses"]
            activity = match["Activity"]
            data_object = match["DataObject"]

            activity_label = node_labels[activity]
            activity_well_formed = self.activity_label_well_formed(activity_label)
            data_object_label = node_labels[data_object]
            data_object_well_formed = self.data_object_well_formed(data_object_label)

            if not activity_well_formed and not data_object_well_formed:
                # both malformed, can't handle this
                print(f"WARN: Skipping Uses '{association}' "
                      f"connecting malformed Activity ('{activity_label}') "
                      f"and malformed DataObject ('{data_object_label}').")
                continue

            if activity_well_formed and data_object_well_formed:
                # both are fine, continue
                continue

            if not activity_well_formed:
                continue

            if not data_object_well_formed:
                data_object_label = self.data_object_from_activity(activity_label)
                node_labels[data_object] = data_object_label
                nx.set_node_attributes(graph, node_labels, "label")



class UnlabeledActorProcessor(BaseProcessor):
    def process(self, graph: nx.DiGraph):
        labels = nx.get_node_attributes(graph, "label")
        types = nx.get_node_attributes(graph, "type")
        num_unlabeled_actors = 0
        for node in graph:
            if types[node] not in ["Actor"]:
                continue
            if len(labels[node]) == 0:
                num_unlabeled_actors += 1
                labels[node] = f"Actor {num_unlabeled_actors}"
        nx.set_node_attributes(graph, labels, "label")


class DeferredChoiceProcessor(BaseProcessor):
    def process(self, graph: nx.DiGraph):
        pattern = nx.DiGraph()
        pattern.add_node("Exclusive", type="Exclusive")
        pattern.add_node("FlowExclusiveOption1", type="Flow")
        pattern.add_node("ActivityOption1", type="Event")
        pattern.add_node("FlowExclusiveOption2", type="Flow")
        pattern.add_node("ActivityOption2", type="Event")

        pattern.add_edge("Exclusive", "FlowExclusiveOption1")
        pattern.add_edge("FlowExclusiveOption1", "ActivityOption1")
        pattern.add_edge("Exclusive", "FlowExclusiveOption2")
        pattern.add_edge("FlowExclusiveOption2", "ActivityOption2")

        matches = patterns.find_graph_pattern(graph, pattern)
        while True:
            try:
                match = next(matches)
            except StopIteration:
                break

            outcomes = patterns.get_successors_of_type(graph, match["Exclusive"], types=["Flow"])

            for outcome in outcomes:
                labels = nx.get_node_attributes(graph, "label")
                events = patterns.get_successors_of_type(graph, outcome, types=["Event"])
                if len(events) == 0:
                    continue
                assert len(events) == 1
                event = events[0]
                event_label = labels[event]

                event_flows = patterns.get_successors_of_type(graph, event, types=["Flow"])
                assert len(event_flows) == 1
                event_flow = event_flows[0]
                flow_targets = graph.out_edges(event_flow)
                assert len(flow_targets) == 1
                flow_target = flow_targets[0]

                # remove event and its outgoing sequence flow
                graph.remove_node(event)
                graph.remove_node(event_flow)

                # repair graph
                labels = nx.get_node_attributes(graph, "label")
                labels[outcome] = event_label
                nx.set_node_attributes(graph, labels, "label")
                graph.add_edge(outcome, flow_target)

            matches = patterns.find_graph_pattern(graph, pattern)


class MessageEventProcessor(BaseProcessor):
    def process(self, graph: nx.DiGraph):
        types = nx.get_node_attributes(graph, "type")
        for node in graph.nodes:
            if types[node] != "MessageEvent":
                continue
            types[node] = "Task"
        nx.set_node_attributes(graph, types, "type")


if __name__ == "__main__":
    def main():
        with open("../resources/plausible/models/0.csv", "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            next(csv_reader)
            model_name = "Event Oranization 1BV00 exam Jane DoeJane Doe 12345678"
            for row in csv_reader:
                if row[6] != model_name:
                    continue
                sam_json = json.loads(row[4])
                print(f"Processing model {row[6]} ({row[8]})...")
                g = conversion.sam_json_to_networkx(sam_json, {
                    "Task": "Activity",
                    "StartNoneEvent": "StartEvent",
                    "StartMessageEvent": "StartEvent",
                    "EndNoneEvent": "EndEvent",
                    "EndTerminateEvent": "EndEvent",
                    "IntermediateTimerEvent": "TimerEvent",
                    "IntermediateMessageEventThrowing": "MessageEvent",
                    "IntermediateMessageEventCatching": "MessageEvent",
                    "SequenceFlow": "Flow",
                    "MessageFlow": "Flow",
                    "ParallelGateway": "Inclusive",
                    "Exclusive_Databased_Gateway": "Exclusive",
                    "Pool": "Actor",
                    "Lane": "Actor",
                    "DataObject": "DataObject",
                    "Association_Unidirectional": "Uses",
                    "Association_Undirected": "Uses"
                })
                conversion.draw_process_graph(g)
                plt.show()
                DataObjectAssociationProcessor().process(g)
                DataObjectLabelProcessor().process(g)
                NounActivityProcessor().process(g)
                conversion.draw_process_graph(g)
                plt.show()
                exit(1)


    main()
