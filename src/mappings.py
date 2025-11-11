import abc
import typing


class MappingCollection(abc.ABC):
    @property
    @abc.abstractmethod
    def ignored(self) -> typing.Set[str]:
        pass

    @property
    @abc.abstractmethod
    def behaviour(self) -> typing.Dict[str, str]:
        pass

    @property
    @abc.abstractmethod
    def data(self) -> typing.Dict[str, str]:
        pass

    @property
    @abc.abstractmethod
    def organization(self) -> typing.Dict[str, str]:
        pass

    @property
    def all(self) -> typing.Dict[str, str]:
        return {
            **self.behaviour,
            **self.data,
            **self.organization
        }


class SimpleSapSamCollection(MappingCollection):
    @property
    def ignored(self) -> typing.Set[str]:
        return set()

    @property
    def behaviour(self) -> typing.Dict[str, str]:
        return {
            "Task": "Activity",
            "CollapsedSubprocess": "Activity",

            "StartCompensationEvent": "StartEvent",
            "StartConditionalEvent": "StartEvent",
            "StartMessageEvent": "StartEvent",
            "StartNoneEvent": "StartEvent",
            "StartParallelMultipleEvent": "StartEvent",
            "StartSignalEvent": "StartEvent",
            "StartTimerEvent": "StartEvent",

            "EndCancelEvent": "EndEvent",
            "EndCompensationEvent": "EndEvent",
            "EndErrorEvent": "EndEvent",
            "EndEscalationEvent": "EndEvent",
            "EndMessageEvent": "EndEvent",
            "EndMultipleEvent": "EndEvent",
            "EndNoneEvent": "EndEvent",
            "EndSignalEvent": "EndEvent",
            "EndTerminateEvent": "EndEvent",

            "Exclusive_Databased_Gateway": "Gateway",
            "ParallelGateway": "Gateway",
        }

    @property
    def data(self) -> typing.Dict[str, str]:
        return {}

    @property
    def organization(self) -> typing.Dict[str, str]:
        return {
            "Lane": "Actor",
            "Pool": "Actor"
        }


class SapSamMappingCollection(MappingCollection):
    @property
    def ignored(self) -> typing.Set[str]:
        return {
            "BPMNDiagram",
            "CollapsedPool",
            "CollapsedVerticalPool",
            "DataStore",
            "ITSystem",
            "TextAnnotation",
            "processparticipant",
            "Message"
        }

    @property
    def disallowed(self) -> typing.Set[str]:
        return {
            "IntermediateEvent",
            "IntermediateCancelEvent",
            "IntermediateTimerEvent",
            "IntermediateErrorEvent",
            "IntermediateMessageEvent",
            "IntermediateConditionalEvent",
            "IntermediateEscalationEvent",
            "IntermediateCompensationEventCatching",
            "IntermediateCompensationEventThrowing",
            "IntermediateMessageEventCatching",
            "IntermediateMessageEventThrowing",
            "IntermediateMultipleEventCatching",
            "IntermediateMultipleEventThrowing",
            "IntermediateSignalEventCatching",
            "IntermediateSignalEventThrowing",
            "IntermediateMessageEventCatching",
            "IntermediateMessageEventThrowing",
            "IntermediateLinkEventCatching",
            "IntermediateLinkEventThrowing",
            "IntermediateEscalationEventCatching",
            "IntermediateEscalationEventThrowing",
            "IntermediateParallelMultipleEventCatching",
            "IntermediateParallelMultipleEventThrowing",
            "Subprocess",
            "Group",
            "CollapsedProcess",
            "CollapsedEventSubprocess",
            "EventSubprocess",
            "EventbasedGateway",
            "InclusiveGateway",
            "ComplexGateway",
            "ChoreographyTask",
            "ChoreographyParticipant"
        }

    @property
    def behaviour(self):
        return {
            "Task": "Activity",
            "CollapsedSubprocess": "Activity",

            "StartMultipleEvent": "StartEvent",
            "StartCompensationEvent": "StartEvent",
            "StartConditionalEvent": "StartEvent",
            "StartErrorEvent": "StartEvent",
            "StartMessageEvent": "StartEvent",
            "StartNoneEvent": "StartEvent",
            "StartParallelMultipleEvent": "StartEvent",
            "StartSignalEvent": "StartEvent",
            "StartTimerEvent": "StartEvent",
            "StartEscalationEvent": "StartEvent",

            "EndCancelEvent": "EndEvent",
            "EndCompensationEvent": "EndEvent",
            "EndErrorEvent": "EndEvent",
            "EndEscalationEvent": "EndEvent",
            "EndMessageEvent": "EndEvent",
            "EndMultipleEvent": "EndEvent",
            "EndNoneEvent": "EndEvent",
            "EndSignalEvent": "EndEvent",
            "EndTerminateEvent": "EndEvent",

            "SequenceFlow": "Flow",
            "MessageFlow": "Flow",

            "ParallelGateway": "Parallel",
            "Exclusive_Databased_Gateway": "Exclusive",
        }

    @property
    def data(self):
        return {
            "DataObject": "DataObject",
            "Association_Unidirectional": "Uses",
            "Association_Undirected": "Uses",
            "Association_Bidirectional": "Uses"
        }

    @property
    def organization(self):
        return {
            "Pool": "Actor",
            "Lane": "Actor",
            "VerticalPool": "Actor",
            "VerticalLane": "Actor",
        }
