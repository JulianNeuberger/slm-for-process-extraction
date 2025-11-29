import csv
import pathlib
import typing

import dotenv
import networkx as nx
import tqdm

import conversion
import load
import mappings
import postprocess
import templating
from templating import util

dotenv.load_dotenv()

resources_folder = pathlib.Path(__file__).parent.parent / "resources"


def post_process_graph(graph: nx.DiGraph):
    postprocessing_pipeline = [
        # postprocess.DataObjectAssociationProcessor(),
        # postprocess.DataObjectLabelProcessor(),
        postprocess.NounActivityProcessor(),
        postprocess.MultiLineLabelProcessor(),
        # postprocess.DeferredChoiceProcessor(),
        postprocess.UnlabeledActorProcessor()
    ]
    for p in postprocessing_pipeline:
        p.process(graph)
    return graph


def apply_fact_templates(graph: nx.DiGraph) -> typing.List[templating.Fact]:
    templates = [
        templating.ActorFactTemplate(),
        templating.TaskFactTemplate(),
    ]
    facts = []
    for t in templates:
        facts.extend(t.generate(graph))
    return facts


def apply_rule_templates(graph: nx.DiGraph, include_tags=True) -> typing.List[templating.Rule]:
    rule_templates = [
        templating.StructuredLoopTemplate(include_tags),
        templating.OptionalRuleTemplate(include_tags),
        templating.ExclusiveChoiceTemplate(include_tags),
        templating.ExplicitMergeTemplate(include_tags),
        templating.ImplicitMergeTemplate(include_tags),
        templating.ParallelSplitTemplate(include_tags),
        templating.SynchronizationTemplate(include_tags),
        templating.InclusiveSplitRuleTemplate(include_tags),
        templating.StructuredSynchronizingMergeRuleTemplate(include_tags),
        templating.SequenceFlowTemplate(include_tags)
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
                resolved = util.resolve_reference(ref, rule_id_by_node, graph, with_tags=include_tags)
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


def generate_sbvr(*, in_file: pathlib.Path, out_file: pathlib.Path):
    if out_file.exists():
        print(f"Already generated SBVR for {in_file.name}. Skipping.")
        return

    models = list(load.load_raw_models(in_file))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        wrote_header = False
        for model in tqdm.tqdm(models):
            mapping = mappings.SapSamMappingCollection()
            g = conversion.sam_json_to_networkx(model.model_json, mapping.all)

            post_process_graph(g)
            rules = apply_rule_templates(g, include_tags=True)
            facts = apply_fact_templates(g)

            unvisited = []
            for node, attr in g.nodes(data=True):
                if attr.get("visited", False):
                    continue
                if attr["type"] not in mapping.behaviour.values():
                    continue
                unvisited.append(f"{attr['label']} ({attr['type']})")
            if len(unvisited) > 0:
                print(f"Unvisited nodes in {model.id}: {unvisited}")

            model_sbvr = load.ModelSBVR(
                model=model,
                sbvr=load.SBVR(
                    rules=[f"R{i}: {r.text}" for i, r in enumerate(rules)],
                    vocab=[f.text for f in facts]
                )
            )

            if not wrote_header:
                writer.writerow(model_sbvr.row.keys())
                wrote_header = True
            writer.writerow(model_sbvr.row.values())


def count_selected_models(*, models_dir: pathlib.Path) -> int:
    total = 0
    for models_file in models_dir.iterdir():
        if models_file.suffix != ".csv":
            continue
        total += len(list(load.load_raw_models(models_file)))
    return total


def main():
    csv.field_size_limit(2147483647)
    resources_dir = pathlib.Path(__file__).parent.parent / "resources"

    total_num_selected = count_selected_models(models_dir=resources_dir / "models" / "selected")
    print(f"Selected {total_num_selected} models!")

    for f in (resources_dir / "models" / "selected").iterdir():
        if f.suffix != ".csv":
            continue
        print(f"Working on file {f.name}")

        # print("\tDrawing models and storing them ...")
        # show.draw_models_from_file(in_file=f,
        #                            image_directory=resources_dir / "images",
        #                            store=False, overwrite=False)
        print("\tGenerating SBVR from models ...")
        generate_sbvr(in_file=f,
                      out_file=(resources_dir / "models" / "sbvr" / f"{f.stem}.csv"))


if __name__ == "__main__":
    main()
