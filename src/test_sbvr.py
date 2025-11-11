import csv
import pathlib
import typing

import networkx as nx

import conversion
import load
import mappings
import postprocess
import templating


def apply_rule_templates(graph: nx.DiGraph) -> typing.List[templating.UnresolvedRule]:
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
        templating.SequenceFlowTemplate(),
        templating.TaskRuleTemplate()
    ]

    rules_by_node: typing.Dict[str, ] = {}
    unresolved_rules: typing.List[templating.UnresolvedRule] = []
    for t in rule_templates:
        for r in t.generate(graph):

            unresolved_rules.append(r)
    unresolved_rules.sort(key=lambda _r: _r.depth)

    rules = []
    for r in unresolved_rules:


        rules.extend(new_rules)
    return rules


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
    print(model.id)
    mapping = mappings.SapSamMappingCollection()
    g = conversion.sam_json_to_networkx(model.model_json, mapping.all)

    post_process_graph(g)
    rules = apply_rule_templates(g)
    rules.sort(key=lambda r: r.depth_in_process)

    unvisited = []
    for node, attr in g.nodes(data=True):
        if attr.get("visited", False):
            continue
        if attr["type"] not in mapping.behaviour.values():
            continue
        unvisited.append(f"{attr['label']} ({attr['type']})")
    # if len(unvisited) > 0:
    #     print(f"Unvisited nodes in {model.id}: {unvisited}")

    print("\n".join([r.text for r in rules]))


def main():
    csv.field_size_limit(2147483647)
    # model = "gpt-5-mini-2025-08-07"
    # model = "gpt-5-nano-2025-08-07"
    # model = "gpt-5-2025-08-07"
    resources_dir = pathlib.Path(__file__).parent.parent / "resources"
    model_id = "0a6881c195c145aa89533984c15e900d"

    for f in (resources_dir / "models" / "selected").iterdir():
        if f.suffix != ".csv":
            continue
        print(f"Working on file {f.name}")
        models = list(load.load_raw_models(f))
        for m in models:
            if m.id != model_id:
                continue


        print("\tDrawing models and storing them ...")
        show.draw_models_from_file(in_file=f,
                                   image_directory=resources_dir / "images",
                                   store=False, overwrite=False)
        print("\tGenerating SBVR from models ...")
        generate_sbvr(in_file=f,
                      out_file=(resources_dir / "models" / "sbvr" / f"{f.stem}.csv"))