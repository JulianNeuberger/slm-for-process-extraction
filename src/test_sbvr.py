import csv
import pathlib
import typing

import networkx as nx

import conversion
import load
import mappings
import postprocess
import templating
from templating import util


def apply_rule_templates(graph: nx.DiGraph) -> typing.List[templating.Rule]:
    rule_templates = [
        templating.StructuredLoopTemplate(),
        templating.OptionalRuleTemplate(),
        templating.ExclusiveChoiceTemplate(),
        templating.ExplicitMergeTemplate(),
        templating.ImplicitMergeTemplate(),
        templating.ParallelSplitTemplate(),
        templating.SynchronizationTemplate(),
        templating.InclusiveSplitRuleTemplate(),
        templating.StructuredSynchronizingMergeRuleTemplate(),
        templating.SequenceFlowTemplate()
    ]

    unresolved_rules: typing.List[templating.UnresolvedRule] = []
    for t in rule_templates:
        for r in t.generate(graph):
            unresolved_rules.append(r)
    unresolved_rules.sort(key=lambda _r: _r.depth)

    rule_id_by_node: typing.Dict[str, int] = {}
    for i, r in enumerate(unresolved_rules):
        for n in r.nodes:
            rule_id_by_node[n] = i

    max_resolve_steps = 20
    while max_resolve_steps > 0:
        max_resolve_steps -= 1
        for r in unresolved_rules:
            refs = [c for c in r.content if isinstance(c, templating.ForwardReference)]
            for ref in refs:
                ref_index = r.content.index(ref)
                resolved = util.resolve_reference(ref, rule_id_by_node, graph)
                # splice in the (possible partially) resolved ref
                r.content = r.content[0:ref_index] + resolved + r.content[ref_index + 1:]

    # resolve all unresolved references to their ids
    for r in unresolved_rules:
        refs = [c for c in r.content if isinstance(c, templating.ForwardReference)]
        for ref in refs:
            ref_index = r.content.index(ref)
            if ref.node not in rule_id_by_node:
                print(f"Reference to node {ref.node} does not belong to any rule")
                print(f"type: {graph.nodes[ref.node]['type']}, label: {graph.nodes[ref.node]['label']}")

            r.content[ref_index] = f"rule {rule_id_by_node[ref.node]}"

    return [
        templating.Rule(
            id=str(i),
            text=" ".join(r.content),
        )
        for i, r in enumerate(unresolved_rules)
    ]


def post_process_graph(graph: nx.DiGraph):
    postprocessing_pipeline = [
        postprocess.NounActivityProcessor(),
        postprocess.MultiLineLabelProcessor(),
        postprocess.UnlabeledActorProcessor()
    ]
    for p in postprocessing_pipeline:
        p.process(graph)
    return graph


def generate_sbvr(model: load.ModelInfo):
    mapping = mappings.SapSamMappingCollection()
    g = conversion.sam_json_to_networkx(model.model_json, mapping.all)

    post_process_graph(g)
    rules = apply_rule_templates(g)

    unvisited = []
    for node, attr in g.nodes(data=True):
        if attr.get("visited", False):
            continue
        if attr["type"] not in mapping.behaviour.values():
            continue
        unvisited.append(f"{attr['label']} ({attr['type']})")
    if len(unvisited) > 0:
        print(f"\t!!! UNVISITED nodes in {model.id}: {unvisited}")

    print("\n".join([f"R{i}: {r.text}" for i, r in enumerate(rules)]))


def main():
    csv.field_size_limit(2147483647)
    # model = "gpt-5-mini-2025-08-07"
    # model = "gpt-5-nano-2025-08-07"
    # model = "gpt-5-2025-08-07"
    resources_dir = pathlib.Path(__file__).parent.parent / "resources"

    for f in (resources_dir / "models" / "selected").iterdir():
        if f.suffix != ".csv":
            continue
        models = list(load.load_raw_models(f))
        for m in models:
            print(m.id + " ----------------------------")
            generate_sbvr(m)
            print("------------------------------------")
            print()

        exit()


if __name__ == "__main__":
    main()
