import csv
import json
import pathlib
import time
import typing
from datetime import datetime

import colorama
import openai
import tqdm
from openai.types import Batch

import annotate
import data
import description
import load
import prompts

resources_folder = pathlib.Path(__file__).parent.parent / "resources"


def batch_line(*, custom_id: str, body: typing.Dict):
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    })


def generate_descriptions_batch(*,
                                in_file: pathlib.Path,
                                out_file: pathlib.Path,
                                example: typing.Optional[str],
                                client: openai.OpenAI,
                                model: str):
    models = list(load.load_sbvr_models(in_file))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    assert out_file.suffix == ".jsonl", "Can only write batches to JSON lines file"

    with open(out_file, "w", encoding="utf-8") as out_f:
        for model_sbvr in tqdm.tqdm(models):
            text = "\n".join(model_sbvr.sbvr.rules)
            sbvr_request = description.LLMSBVRDescriber(client, model).request_params(sbvr=text,
                                                                                      image_path=None,
                                                                                      example=example)

            image_path = resources_folder / "images" / f"{model_sbvr.model.id}.png"
            image_request = description.LLMPictureDescriber(client, model).request_params(sbvr=None,
                                                                                          image_path=image_path,
                                                                                          example=example)

            image_path = resources_folder / "images" / f"{model_sbvr.model.id}.png"
            combined_request = description.LLMCombinedDescriber(client, model).request_params(sbvr=text,
                                                                                              image_path=image_path,
                                                                                              example=example)

            out_f.write(batch_line(custom_id=f"describe-{model_sbvr.model.id}-sbvr", body=sbvr_request) + "\n")
            out_f.write(batch_line(custom_id=f"describe-{model_sbvr.model.id}-image", body=image_request) + "\n")
            out_f.write(batch_line(custom_id=f"describe-{model_sbvr.model.id}-combined", body=combined_request) + "\n")


def write_batch_answers(batch_info_paths: typing.List[pathlib.Path], client: openai.OpenAI, ):
    for batch_info_path in batch_info_paths:
        with open(batch_info_path, "r", encoding="utf-8") as f:
            batch_info = Batch.model_validate_json(f.read())
        assert batch_info.status == "completed"
        batch_answer_file_path = batch_info_path.parent.parent / "outputs" / f"{batch_info_path.stem}.jsonl"
        if batch_answer_file_path.exists():
            continue
        batch_answer_file_path.parent.mkdir(parents=True, exist_ok=True)
        batch_answers = client.files.content(batch_info.output_file_id)
        with open(batch_answer_file_path, "w", encoding="utf-8") as f:
            f.write(batch_answers.text)


def load_answers_by_model_id(answers_file_path: pathlib.Path) -> typing.Dict[str, typing.Dict[str, load.LLMCompletion]]:
    answers_by_model_id = {}
    with (open(answers_file_path, "r", encoding="utf-8") as f):
        for line in f:
            if line.strip() == "":
                continue
            data = json.loads(line)
            # e.g. describe-1d1451c4a6e9488ba78e34cee314cf10-sbvr
            custom_id = data["custom_id"]
            _, model_id, hints = custom_id.split("-")
            if model_id not in answers_by_model_id:
                answers_by_model_id[model_id] = {}
            body = data["response"]["body"]
            answers_by_model_id[model_id][hints] = load.LLMCompletion(
                text=body["choices"][0]["message"]["content"],
                prompt_tokens=body["usage"]["prompt_tokens"],
                completion_tokens=body["usage"]["completion_tokens"],
            )
    return answers_by_model_id


def write_described(*,
                    models_sbvr_file_path: pathlib.Path,
                    answers_file_path: pathlib.Path,
                    output_file_path: pathlib.Path, ):
    answers_by_model_id = load_answers_by_model_id(answers_file_path)

    wrote_header = False
    with open(output_file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for model_sbvr in load.load_sbvr_models(models_sbvr_file_path):
            described_model = load.DescribedModel(
                model=model_sbvr.model,
                sbvr=model_sbvr.sbvr,
                descriptions=load.ProcessDescriptions(
                    from_sbvr=answers_by_model_id[model_sbvr.model.id]["sbvr"],
                    from_picture=answers_by_model_id[model_sbvr.model.id]["image"],
                    from_both=answers_by_model_id[model_sbvr.model.id]["combined"]
                )
            )
            if not wrote_header:
                writer.writerow(described_model.row.keys())
                wrote_header = True
            writer.writerow(described_model.row.values())


def generate_mention_annotations_batch(*,
                                       in_file: pathlib.Path,
                                       out_file: pathlib.Path,
                                       client: openai.OpenAI,
                                       model: str):
    models = list(load.load_described_models(in_file))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    hint_template_path = resources_folder / "prompts" / "hints" / "sbvr.txt"
    hint_template = prompts.Prompt(hint_template_path)

    assert out_file.suffix == ".jsonl", "Can only write batches to JSON lines file"

    with open(out_file, "w", encoding="utf-8") as out_f:
        for described_model in tqdm.tqdm(models):
            image_path = resources_folder / "images" / f"{described_model.model.id}.png"
            rules = "\n".join(described_model.sbvr.rules)
            facts = "\n".join(described_model.sbvr.vocab)
            assert rules is not None and len(rules) > 0
            assert facts is not None and len(facts) > 0

            hint = hint_template(rules=rules, vocabulary=facts)

            annotator = annotate.LLMMentionAnnotator(client, model, reasoning_effort="low")

            from_sbvr_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_sbvr.text,
                                                           described_model.model.id)
            sbvr_annotation = annotator.batch_line(doc=from_sbvr_doc, hints=hint, image_path=None)
            out_f.write(json.dumps(sbvr_annotation) + "\n")

            from_image_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_picture.text,
                                                            described_model.model.id)
            image_annotation = annotator.batch_line(doc=from_image_doc, hints=None, image_path=image_path)
            out_f.write(json.dumps(image_annotation) + "\n")

            from_combined_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_both.text,
                                                               described_model.model.id)
            combined_annotation = annotator.batch_line(doc=from_combined_doc, hints=hint, image_path=image_path)
            out_f.write(json.dumps(combined_annotation) + "\n")


def generate_entity_annotations_batch(*,
                                      in_file: pathlib.Path,
                                      mention_answers_file: pathlib.Path,
                                      out_file: pathlib.Path,
                                      client: openai.OpenAI,
                                      model: str):
    models = list(load.load_described_models(in_file))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mention_answers_by_id = load_answers_by_model_id(mention_answers_file)

    hint_template_path = resources_folder / "prompts" / "hints" / "sbvr.txt"
    hint_template = prompts.Prompt(hint_template_path)

    assert out_file.suffix == ".jsonl", "Can only write batches to JSON lines file"

    with open(out_file, "w", encoding="utf-8") as out_f:
        for described_model in tqdm.tqdm(models):
            image_path = resources_folder / "images" / f"{described_model.model.id}.png"
            rules = "\n".join(described_model.sbvr.rules)
            facts = "\n".join(described_model.sbvr.vocab)
            assert rules is not None and len(rules) > 0
            assert facts is not None and len(facts) > 0

            hint = hint_template(rules=rules, vocabulary=facts)

            mentions_annotator = annotate.LLMMentionAnnotator(client, model, reasoning_effort="low")
            entities_annotator = annotate.LLMEntitiesAnnotator(client, model, reasoning_effort="low")

            mention_answers = mention_answers_by_id[described_model.model.id]

            from_sbvr_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_sbvr.text,
                                                           described_model.model.id)
            from_sbvr_doc = mentions_annotator.parser.parse(document=from_sbvr_doc,
                                                            string=mention_answers["sbvr"].text)
            sbvr_annotation = entities_annotator.batch_line(doc=from_sbvr_doc, hints=hint, image_path=None)
            out_f.write(json.dumps(sbvr_annotation) + "\n")

            from_image_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_picture.text,
                                                            described_model.model.id)
            from_image_doc = mentions_annotator.parser.parse(document=from_image_doc,
                                                             string=mention_answers["image"].text)
            image_annotation = entities_annotator.batch_line(doc=from_image_doc, hints=None, image_path=image_path)
            out_f.write(json.dumps(image_annotation) + "\n")

            from_combined_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_both.text,
                                                               described_model.model.id)
            from_combined_doc = mentions_annotator.parser.parse(document=from_combined_doc,
                                                                string=mention_answers["combined"].text)
            combined_annotation = entities_annotator.batch_line(doc=from_combined_doc, hints=hint,
                                                                image_path=image_path)
            out_f.write(json.dumps(combined_annotation) + "\n")


def generate_relation_annotations_batch(*,
                                        in_file: pathlib.Path,
                                        mention_answers_file: pathlib.Path,
                                        entities_answers_file: pathlib.Path,
                                        out_file: pathlib.Path,
                                        client: openai.OpenAI,
                                        model: str):
    models = list(load.load_described_models(in_file))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mention_answers_by_id = load_answers_by_model_id(mention_answers_file)
    entity_answers_by_id = load_answers_by_model_id(entities_answers_file)

    hint_template_path = resources_folder / "prompts" / "hints" / "sbvr.txt"
    hint_template = prompts.Prompt(hint_template_path)

    assert out_file.suffix == ".jsonl", "Can only write batches to JSON lines file"

    with open(out_file, "w", encoding="utf-8") as out_f:
        for described_model in tqdm.tqdm(models):
            image_path = resources_folder / "images" / f"{described_model.model.id}.png"
            rules = "\n".join(described_model.sbvr.rules)
            facts = "\n".join(described_model.sbvr.vocab)
            assert rules is not None and len(rules) > 0
            assert facts is not None and len(facts) > 0

            hint = hint_template(rules=rules, vocabulary=facts)

            mentions_annotator = annotate.LLMMentionAnnotator(client, model, reasoning_effort="low")
            entities_annotator = annotate.LLMEntitiesAnnotator(client, model, reasoning_effort="low")
            relations_annotator = annotate.LLMRelationsAnnotator(client, model, reasoning_effort="low")

            mention_answers = mention_answers_by_id[described_model.model.id]
            entity_answers = entity_answers_by_id[described_model.model.id]

            from_sbvr_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_sbvr.text,
                                                           described_model.model.id)
            from_sbvr_doc = mentions_annotator.parser.parse(document=from_sbvr_doc,
                                                            string=mention_answers["sbvr"].text)
            from_sbvr_doc = entities_annotator.parser.parse(document=from_sbvr_doc,
                                                            string=entity_answers["sbvr"].text)
            sbvr_annotation = relations_annotator.batch_line(doc=from_sbvr_doc, hints=hint, image_path=None)
            out_f.write(json.dumps(sbvr_annotation) + "\n")

            from_image_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_picture.text,
                                                            described_model.model.id)
            from_image_doc = mentions_annotator.parser.parse(document=from_image_doc,
                                                             string=mention_answers["image"].text)
            from_image_doc = entities_annotator.parser.parse(document=from_image_doc,
                                                             string=entity_answers["image"].text)
            image_annotation = relations_annotator.batch_line(doc=from_image_doc, hints=None, image_path=image_path)
            out_f.write(json.dumps(image_annotation) + "\n")

            from_combined_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_both.text,
                                                               described_model.model.id)
            from_combined_doc = mentions_annotator.parser.parse(document=from_combined_doc,
                                                                string=mention_answers["combined"].text)
            from_combined_doc = entities_annotator.parser.parse(document=from_combined_doc,
                                                                string=entity_answers["combined"].text)
            combined_annotation = relations_annotator.batch_line(doc=from_combined_doc, hints=hint,
                                                                 image_path=image_path)
            out_f.write(json.dumps(combined_annotation) + "\n")


def count_selected_models(*, models_dir: pathlib.Path) -> int:
    total = 0
    for models_file in models_dir.iterdir():
        if models_file.suffix != ".csv":
            continue
        total += len(list(load.load_raw_models(models_file)))
    return total


def upload_batch(*, batch_file_path: pathlib.Path, client: openai.OpenAI) -> str:
    id_file_path = batch_file_path.parent.parent / "ids" / f"{batch_file_path.stem}.fileid"
    if id_file_path.exists():
        print(f"\tBatch {batch_file_path.name} already uploaded, skipping.")
        return id_file_path.read_text()
    batch_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    id_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(id_file_path, "w") as f:
        f.write(batch_file.id)
    return batch_file.id


def start_batch(*, batch_file_path: pathlib.Path, client: openai.OpenAI) -> Batch:
    batch_id = upload_batch(batch_file_path=batch_file_path, client=client)
    batch_info_file_path = batch_file_path.parent.parent / "infos" / f"{batch_file_path.stem}.json"

    if batch_info_file_path.exists():
        with open(batch_info_file_path, "r") as f:
            batch = Batch.model_validate_json(f.read())
        print(f"Batch already started, status: {batch.status}.")
        return batch

    batch = client.batches.create(
        input_file_id=batch_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_info_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_info_file_path, "w") as f:
        f.write(batch.model_dump_json())
    return batch


def check_batch_status(*, batch_info_file_path: pathlib.Path, client: openai.OpenAI) -> str:
    with open(batch_info_file_path, "r") as f:
        batch = Batch.model_validate_json(f.read())
    batch = client.batches.retrieve(batch.id)
    with open(batch_info_file_path, "w") as f:
        f.write(batch.model_dump_json())
    return batch.status


def wait_for_batches(batch_info_file_paths: typing.List[pathlib.Path], client: openai.OpenAI) -> None:
    symbols = {
        "validating": f"{colorama.Fore.YELLOW}▁",
        "in_progress": f"{colorama.Fore.YELLOW}▄",
        "finalizing": f"{colorama.Fore.YELLOW}▆",
        "completed": f"{colorama.Fore.GREEN}█",

        "expired": f"{colorama.Fore.RED}▓",
        "failed": f"{colorama.Fore.RED}▓",
        "cancelling": f"{colorama.Fore.YELLOW}░",
        "cancelled": f"{colorama.Fore.RED}▓"
    }

    terminated = [
        "failed", "expired", "cancelled", "completed"
    ]

    finished = False
    print()
    while not finished:
        status_string = f"Status of {len(batch_info_file_paths)} batches: "
        finished = True
        for f in batch_info_file_paths:
            status = check_batch_status(batch_info_file_path=f, client=client)
            status_string += symbols[status]
            if status not in terminated:
                finished = False
        print(f"\r[{datetime.now():%Y-%m-%d %H:%M:%S}] {status_string}{colorama.Style.RESET_ALL}", end="", flush=True)
        if finished:
            print()
            break
        time.sleep(5)


def models_from_answers(*,
                        client: openai.OpenAI,
                        model: str,
                        model_path: pathlib.Path,
                        mention_answers_path: pathlib.Path,
                        entities_answers_path: pathlib.Path,
                        relations_answers_path: pathlib.Path,
                        out_directory: pathlib.Path):
    models = list(load.load_described_models(model_path))
    out_directory.parent.mkdir(parents=True, exist_ok=True)

    mention_answers_by_id = load_answers_by_model_id(mention_answers_path)
    entity_answers_by_id = load_answers_by_model_id(entities_answers_path)
    relation_answers_by_id = load_answers_by_model_id(relations_answers_path)

    from_sbvr_out_file = out_directory / "sbvr" / f"{model_path.stem}.jsonl"
    from_sbvr_out_file.parent.mkdir(parents=True, exist_ok=True)
    from_image_out_file = out_directory / "image" / f"{model_path.stem}.jsonl"
    from_image_out_file.parent.mkdir(parents=True, exist_ok=True)
    from_combined_out_file = out_directory / "combined" / f"{model_path.stem}.jsonl"
    from_combined_out_file.parent.mkdir(parents=True, exist_ok=True)

    with (open(from_sbvr_out_file, "w") as sbvr_out,
          open(from_image_out_file, "w") as image_out,
          open(from_combined_out_file, "w") as combined_out):
        for described_model in tqdm.tqdm(models):
            mentions_annotator = annotate.LLMMentionAnnotator(client, model)
            entities_annotator = annotate.LLMEntitiesAnnotator(client, model)
            relations_annotator = annotate.LLMRelationsAnnotator(client, model)

            mention_answers = mention_answers_by_id[described_model.model.id]
            entity_answers = entity_answers_by_id[described_model.model.id]
            relation_answers = relation_answers_by_id[described_model.model.id]

            from_sbvr_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_sbvr.text,
                                                           described_model.model.id)
            from_sbvr_doc = mentions_annotator.parser.parse(document=from_sbvr_doc,
                                                            string=mention_answers["sbvr"].text)
            from_sbvr_doc = entities_annotator.parser.parse(document=from_sbvr_doc,
                                                            string=entity_answers["sbvr"].text)
            from_sbvr_doc = relations_annotator.parser.parse(document=from_sbvr_doc,
                                                             string=relation_answers["sbvr"].text)
            sbvr_out.write(json.dumps(data.PetDictExporter().export_document(from_sbvr_doc)) + "\n")

            from_image_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_picture.text,
                                                            described_model.model.id)
            from_image_doc = mentions_annotator.parser.parse(document=from_image_doc,
                                                             string=mention_answers["image"].text)
            from_image_doc = entities_annotator.parser.parse(document=from_image_doc,
                                                             string=entity_answers["image"].text)
            from_image_doc = relations_annotator.parser.parse(document=from_image_doc,
                                                              string=relation_answers["image"].text)
            image_out.write(json.dumps(data.PetDictExporter().export_document(from_image_doc)) + "\n")

            from_combined_doc = annotate.parse_text_to_pet_doc(described_model.descriptions.from_both.text,
                                                               described_model.model.id)
            from_combined_doc = mentions_annotator.parser.parse(document=from_combined_doc,
                                                                string=mention_answers["combined"].text)
            from_combined_doc = entities_annotator.parser.parse(document=from_combined_doc,
                                                                string=entity_answers["combined"].text)
            from_combined_doc = relations_annotator.parser.parse(document=from_combined_doc,
                                                                 string=relation_answers["combined"].text)
            combined_out.write(json.dumps(data.PetDictExporter().export_document(from_combined_doc)) + "\n")


def main():
    csv.field_size_limit(2147483647)
    # model = "gpt-5-mini-2025-08-07"
    # model = "gpt-5-nano-2025-08-07"
    # model = "gpt-5-2025-08-07"
    resources_dir = pathlib.Path(__file__).parent.parent / "resources"

    client = openai.OpenAI()

    total_num_selected = count_selected_models(models_dir=resources_dir / "models" / "selected")
    print(f"Selected {total_num_selected} models!")

    with open(resources_dir / "prompts" / "examples" / "pet-example.txt") as f:
        example = f.read()

    print("Generating batches for process description ...")
    max_files = 2
    for f in tqdm.tqdm((resources_dir / "models" / "sbvr").iterdir()):
        batch_file_path = (resources_dir / "batches" / "descriptions" / "inputs" / f"{f.stem}.jsonl")
        batch_file_path.parent.mkdir(parents=True, exist_ok=True)
        if batch_file_path.exists():
            pass
        generate_descriptions_batch(in_file=f,
                                    out_file=batch_file_path,
                                    client=client,
                                    model="gpt-5-nano-2025-08-07",
                                    example=example)

        max_files -= 1
        if max_files <= 0:
            # TODO: run entire folder if everything looks fine
            break

    print("Uploading batches ...")
    for batch_file_path in (resources_dir / "batches" / "descriptions" / "inputs").iterdir():
        start_batch(batch_file_path=batch_file_path, client=client)

    print("Waiting for batch completion ...")
    description_batch_info_paths = list((resources_dir / "batches" / "descriptions" / "infos").iterdir())
    wait_for_batches(description_batch_info_paths, client=client)
    write_batch_answers(description_batch_info_paths, client=client)

    print("Batches completed, writing csv ...")
    for f in tqdm.tqdm((resources_dir / "batches" / "descriptions" / "outputs").iterdir()):
        write_described(
            models_sbvr_file_path=resources_dir / "models" / "sbvr" / f"{f.stem}.csv",
            answers_file_path=f,
            output_file_path=resources_dir / "models" / "described" / f"{f.stem}.csv"
        )

    print("Building mention annotation batches ...")
    for f in tqdm.tqdm((resources_dir / "models" / "described").iterdir()):
        out_f = resources_dir / "batches" / "mentions" / "inputs" / f"{f.stem}.jsonl"
        generate_mention_annotations_batch(in_file=f, out_file=out_f,
                                           client=client, model="gpt-5-mini-2025-08-07")

    print("Uploading batches ...")
    batch_file_path = resources_dir / "batches" / "mentions" / "inputs"
    for f in tqdm.tqdm(batch_file_path.iterdir()):
        start_batch(batch_file_path=f, client=client)

    print("Waiting for batch completion ...")
    mention_batches = list((resources_dir / "batches" / "mentions" / "infos").iterdir())
    wait_for_batches(mention_batches, client=client)
    write_batch_answers(mention_batches, client=client)

    print("Building entity annotation batches ...")
    for f in tqdm.tqdm((resources_dir / "models" / "described").iterdir()):
        out_f = resources_dir / "batches" / "entities" / "inputs" / f"{f.stem}.jsonl"
        mentions_answer_f = resources_dir / "batches" / "mentions" / "outputs" / f"{f.stem}.jsonl"
        generate_entity_annotations_batch(in_file=f, out_file=out_f, mention_answers_file=mentions_answer_f,
                                          client=client, model="gpt-5-mini-2025-08-07")
    print("Uploading batches ...")
    batch_file_path = resources_dir / "batches" / "entities" / "inputs"
    for f in tqdm.tqdm(batch_file_path.iterdir()):
        start_batch(batch_file_path=f, client=client)

    print("Waiting for batch completion ...")
    entity_batches = list((resources_dir / "batches" / "entities" / "infos").iterdir())
    wait_for_batches(entity_batches, client=client)
    write_batch_answers(entity_batches, client=client)

    print("Building relation annotation batches ...")
    for f in tqdm.tqdm((resources_dir / "models" / "described").iterdir()):
        out_f = resources_dir / "batches" / "relations" / "inputs" / f"{f.stem}.jsonl"
        mentions_answer_f = resources_dir / "batches" / "mentions" / "outputs" / f"{f.stem}.jsonl"
        entities_answer_f = resources_dir / "batches" / "entities" / "outputs" / f"{f.stem}.jsonl"
        generate_relation_annotations_batch(in_file=f, out_file=out_f,
                                            mention_answers_file=mentions_answer_f,
                                            entities_answers_file=entities_answer_f,
                                            client=client, model="gpt-5-mini-2025-08-07")

    print("Uploading batches ...")
    batch_file_path = resources_dir / "batches" / "relations" / "inputs"
    for f in tqdm.tqdm(batch_file_path.iterdir()):
        start_batch(batch_file_path=f, client=client)

    print("Waiting for batch completion ...")
    relation_batches = list((resources_dir / "batches" / "relations" / "infos").iterdir())
    wait_for_batches(relation_batches, client=client)
    write_batch_answers(relation_batches, client=client)

    print("Dumping generated documents ...")
    for f in tqdm.tqdm((resources_dir / "models" / "described").iterdir()):
        models_from_answers(
            client=client,
            model="gpt-5-nano-2025-08-07",
            model_path=f,
            mention_answers_path=resources_dir / "batches" / "mentions" / "outputs" / f"{f.stem}.jsonl",
            entities_answers_path=resources_dir / "batches" / "entities" / "outputs" / f"{f.stem}.jsonl",
            relations_answers_path=resources_dir / "batches" / "relations" / "outputs" / f"{f.stem}.jsonl",
            out_directory=resources_dir / "docs"
        )


if __name__ == "__main__":
    main()
