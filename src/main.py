import csv
import json
import pathlib
import random
import typing

import dotenv
import networkx as nx
import openai
import tqdm

import annotate
import conversion
import data
import description
import load
import mappings
import postprocess
import prompts
import show
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
            rules = apply_rule_templates(g)

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
                    vocab=[]
                )
            )

            if not wrote_header:
                writer.writerow(model_sbvr.row.keys())
                wrote_header = True
            writer.writerow(model_sbvr.row.values())


def generate_descriptions(*,
                          in_file: pathlib.Path,
                          out_file: pathlib.Path,
                          example: typing.Optional[str],
                          client: openai.OpenAI,
                          model: str,
                          ratio: typing.Optional[float] = None):
    if ratio is None:
        ratio = 1.0
    assert 0 <= ratio <= 1

    models = list(load.load_sbvr_models(in_file))
    indices = list(range(len(models)))
    random.shuffle(indices)
    models = [models[i] for i in indices]

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as out_f:
        wrote_header = False

        writer = csv.writer(out_f, lineterminator="\n")

        for model_sbvr in models:
            print(f"Generating descriptions by rephrasing sbvr rules for model {model_sbvr.model.name} ...")
            text = "\n".join(model_sbvr.sbvr.rules)
            rephrased = description.LLMSBVRDescriber(client, model).describe(sbvr=text,
                                                                             image_path=None,
                                                                             example=example)
            assert len(rephrased.text) > 0

            print(f"Generating descriptions by describing a picture for model {model_sbvr.model.name} ...")
            image_path = resources_folder / "images" / f"{model_sbvr.model.id}.png"
            described = description.LLMPictureDescriber(client, model).describe(sbvr=None,
                                                                                image_path=image_path,
                                                                                example=example)
            assert len(described.text) > 0

            print(f"Generating descriptions by describing a picture and SBVR for model {model_sbvr.model.name} ...")
            image_path = resources_folder / "images" / f"{model_sbvr.model.id}.png"
            combined = description.LLMCombinedDescriber(client, model).describe(sbvr=text,
                                                                                image_path=image_path,
                                                                                example=example)
            assert len(combined.text) > 0

            described_model = load.DescribedModel(
                model=model_sbvr.model,
                sbvr=model_sbvr.sbvr,
                descriptions=load.ProcessDescriptions(
                    from_sbvr=rephrased,
                    from_picture=described,
                    from_both=combined
                )
            )

            if not wrote_header:
                writer.writerow(described_model.row.keys())
                wrote_header = True
            writer.writerow(described_model.row.values())


def annotate_document(document: data.PetDocument,
                      client: openai.OpenAI,
                      model: str,
                      image_path: typing.Optional[pathlib.Path],
                      sbvr_hints: typing.Optional[str]) -> annotate.LLMAnnotation:
    prompt_tokens = 0
    output_tokens = 0

    # mentions
    print("\t\tAnnotating mentions ...")
    annotator = annotate.LLMMentionAnnotator(client, model)
    annotation = annotator.annotate(doc=document, hints=sbvr_hints, image_path=image_path)
    document = annotation.doc
    prompt_tokens += annotation.prompt_tokens
    output_tokens += annotation.completion_tokens
    # entities
    print("\t\tAnnotating entities ...")
    annotator = annotate.LLMEntitiesAnnotator(client, model)
    annotation = annotator.annotate(doc=document, hints=sbvr_hints, image_path=image_path)
    document = annotation.doc
    prompt_tokens += annotation.prompt_tokens
    output_tokens += annotation.completion_tokens
    # relations
    print("\t\tAnnotating relations ...")
    annotator = annotate.LLMRelationsAnnotator(client, model)
    annotation = annotator.annotate(doc=document, hints=sbvr_hints, image_path=image_path)
    document = annotation.doc
    prompt_tokens += annotation.prompt_tokens
    output_tokens += annotation.completion_tokens

    return annotate.LLMAnnotation(
        doc=document,
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
    )


def annotate_descriptions(*,
                          in_file: pathlib.Path,
                          out_file: pathlib.Path,
                          client: openai.OpenAI,
                          model: str):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    exporter = data.PetDictExporter()

    hint_template_path = resources_folder / "prompts" / "hints" / "sbvr.txt"
    hint_template = prompts.Prompt(hint_template_path)

    models = load.load_described_models(in_file)

    with open(out_file, "w", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, lineterminator="\n")
        wrote_header = False

        for described_model in models:
            print(f"Annotating descriptions with SBVR hints for model {described_model.model.name} ...")
            model_id = described_model.model.id
            image_path = resources_folder / "images" / f"{model_id}.png"

            sbvr_description = described_model.descriptions.from_sbvr
            picture_description = described_model.descriptions.from_picture
            combined_description = described_model.descriptions.from_both
            rules = "\n".join(described_model.sbvr.rules)
            facts = "\n".join(described_model.sbvr.vocab)
            assert sbvr_description is not None and len(sbvr_description.text) > 0
            assert rules is not None and len(rules) > 0
            assert facts is not None and len(facts) > 0

            hint = hint_template(rules=rules, vocabulary=facts)

            print("\tAnnotating without any hints ...")
            no_hints_annotation = annotate_document(
                annotate.parse_text_to_pet_doc(sbvr_description.text, model_id),
                client,
                model,
                image_path=None,
                sbvr_hints=None,
            )
            no_hints_doc_as_json = json.dumps(exporter.export_document(no_hints_annotation.doc))

            print("\tAnnotating with SBVR hints ...")
            sbvr_annotation = annotate_document(
                annotate.parse_text_to_pet_doc(sbvr_description.text, model_id),
                client,
                model,
                image_path=None,
                sbvr_hints=hint,
            )
            sbvr_doc_as_json = json.dumps(exporter.export_document(sbvr_annotation.doc))

            print("\tAnnotating with picture hints ...")
            picture_annotation = annotate_document(
                annotate.parse_text_to_pet_doc(picture_description.text, model_id),
                client,
                model,
                image_path=image_path,
                sbvr_hints=None,
            )
            picture_doc_as_json = json.dumps(exporter.export_document(picture_annotation.doc))

            print("\tAnnotating with combined hints ...")
            combined_annotation = annotate_document(
                annotate.parse_text_to_pet_doc(combined_description.text, model_id),
                client,
                model,
                image_path=image_path,
                sbvr_hints=hint,
            )
            combined_doc_as_json = json.dumps(exporter.export_document(combined_annotation.doc))

            annotated_model = load.AnnotatedModel(
                model=described_model.model,
                sbvr=described_model.sbvr,
                descriptions=described_model.descriptions,
                annotations=load.Annotations(
                    no_hints=load.LLMCompletion(
                        text=no_hints_doc_as_json,
                        prompt_tokens=no_hints_annotation.prompt_tokens,
                        completion_tokens=no_hints_annotation.completion_tokens,
                    ),
                    sbvr_hints=load.LLMCompletion(
                        text=sbvr_doc_as_json,
                        prompt_tokens=sbvr_annotation.prompt_tokens,
                        completion_tokens=sbvr_annotation.completion_tokens,
                    ),
                    picture_hints=load.LLMCompletion(
                        text=picture_doc_as_json,
                        prompt_tokens=picture_annotation.prompt_tokens,
                        completion_tokens=picture_annotation.completion_tokens,
                    ),
                    combined_hints=load.LLMCompletion(
                        text=combined_doc_as_json,
                        prompt_tokens=combined_annotation.prompt_tokens,
                        completion_tokens=combined_annotation.completion_tokens,
                    ),
                )
            )

            if not wrote_header:
                writer.writerow(annotated_model.row.keys())
                wrote_header = True
            writer.writerow(annotated_model.row.values())

            break


def count_selected_models(*, models_dir: pathlib.Path) -> int:
    total = 0
    for models_file in models_dir.iterdir():
        if models_file.suffix != ".csv":
            continue
        total += len(list(load.load_raw_models(models_file)))
    return total


def main():
    csv.field_size_limit(2147483647)
    # model = "gpt-5-mini-2025-08-07"
    # model = "gpt-5-nano-2025-08-07"
    # model = "gpt-5-2025-08-07"
    resources_dir = pathlib.Path(__file__).parent.parent / "resources"

    total_num_selected = count_selected_models(models_dir=resources_dir / "models" / "selected")
    print(f"Selected {total_num_selected} models!")

    for f in (resources_dir / "models" / "selected").iterdir():
        if f.suffix != ".csv":
            continue
        print(f"Working on file {f.name}")

        print("\tDrawing models and storing them ...")
        show.draw_models_from_file(in_file=f,
                                   image_directory=resources_dir / "images",
                                   store=False, overwrite=False)
        print("\tGenerating SBVR from models ...")
        generate_sbvr(in_file=f,
                      out_file=(resources_dir / "models" / "sbvr" / f"{f.stem}.csv"))

        # with open(resources_dir / "prompts" / "examples" / "pet-example.txt") as f:
        #     example = f.read()
        #
        # # print("Using an LLM to generate process descriptions ...")
        # # generate_descriptions(in_file=(resources_dir / "models" / "sbvr" / "0.csv"),
        # #                       out_file=(resources_dir / "models" / "described" / "0.csv"),
        # #                       client=openai.OpenAI(),
        # #                       model="gpt-5-nano-2025-08-07",
        # #                       example=example)
        #
        # print("Using an LLM to annotate the descriptions ...")
        # annotate_descriptions(in_file=(resources_dir / "models" / "described" / "0.csv"),
        #                       out_file=(resources_dir / "models" / "annotated" / "0.csv"),
        #                       client=openai.OpenAI(),
        #                       model="gpt-5-mini-2025-08-07")


if __name__ == "__main__":
    main()
