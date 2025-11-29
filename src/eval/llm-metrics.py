import json
import pathlib

import numpy as np

from data.pet import PetImporter
from eval.metrics import Stats


def load_ner_metrics(result_dir: pathlib.Path, original_file: pathlib.Path):
    ground_truth_importer = PetImporter(original_file)
    ground_truth_documents_dict = {
        doc.id: doc for doc in ground_truth_importer.do_import()
    }

    # create importers and import documents
    predictions_files = [f for f in result_dir.iterdir() if f.suffix == '.jsonl']
    assert len(predictions_files) == 1, f"No prediction in {result_dir}"
    prediction_importer = PetImporter(predictions_files[0])
    prediction_documents_dict = {
        doc.id: doc for doc in prediction_importer.do_import()
    }

    # process all the documents
    ner_stats = []
    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        if doc_id not in prediction_documents_dict:
            print(f"Skipping document {doc_id}, as we did not predict that yet.")
            continue

        predicted_doc = prediction_documents_dict[doc_id]

        original_entities = ground_truth_doc.mentions
        original_entities_buffer = original_entities.copy()

        predicted_entities = prediction_documents_dict[doc_id].mentions

        # now collect relation hashes
        matches: int = 0
        for p_entity in predicted_entities:
            p_tokens = set(
                predicted_doc.tokens[i]
                for i
                in p_entity.token_document_indices
            )

            match_found: bool = False
            for idx, t_entity in enumerate(original_entities_buffer):
                t_tokens = set(
                    ground_truth_doc.tokens[i]
                    for i
                    in t_entity.token_document_indices
                )

                if t_entity.type != p_entity.type:
                    # no type match
                    continue

                if not t_tokens.intersection(p_tokens):
                    # head argument is different
                    continue

                # got a match
                match_found = True
                break

            # check match and remove relation from the buffer
            if match_found:
                matches += 1
                original_entities_buffer.remove(t_entity)

        # how we can check the intersection of these two sets and find the results
        ner_stats.append(Stats(num_pred=len(predicted_entities), num_gold=len(original_entities), num_ok=matches))

    times = []
    stats_dir = result_dir / "stats"
    if stats_dir.exists():
        for stats_path in stats_dir.iterdir():
            with open(stats_path, "r") as stats_file:
                stats = json.load(stats_file)
            times.append(stats["duration_seconds"])

    return ner_stats, times


def load_re_metrics(result_dir: pathlib.Path, original_file: pathlib.Path):
    ground_truth_importer = PetImporter(original_file)
    ground_truth_documents_dict = {
        doc.id: doc for doc in ground_truth_importer.do_import()
    }

    # create importers and import documents
    prediction_importer = PetImporter(result_dir / "predictions.jsonl")
    prediction_documents_dict = {
        doc.id: doc for doc in prediction_importer.do_import()
    }

    # process all the documents
    re_stats = []
    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        if doc_id not in prediction_documents_dict:
            # print(f"Skipping document {doc_id}, as we did not predict that yet.")
            continue

        predicted_doc = prediction_documents_dict[doc_id]

        # get original relations
        original_relations = ground_truth_doc.relations
        original_relations_buffer = original_relations.copy()

        # predicted releations
        predicted_relations = prediction_documents_dict[doc_id].relations

        # now collect relation hashes
        matches: int = 0
        for relation in predicted_relations:
            p_head_tokens = set(predicted_doc.tokens[i]
                                for i
                                in predicted_doc.mentions[relation.head_mention_index].token_document_indices)
            p_tail_tokens = set(predicted_doc.tokens[i]
                                for i
                                in predicted_doc.mentions[relation.tail_mention_index].token_document_indices)

            match_found: bool = False
            # look for relation in
            for idx, original_relation in enumerate(original_relations_buffer):
                t_head_tokens = set(ground_truth_doc.tokens[i]
                                    for i
                                    in ground_truth_doc.mentions[
                                        original_relation.head_mention_index].token_document_indices)
                t_tail_tokens = set(ground_truth_doc.tokens[i]
                                    for i
                                    in ground_truth_doc.mentions[
                                        original_relation.tail_mention_index].token_document_indices)

                if original_relation.type != relation.type:
                    # no type match
                    continue

                if not t_head_tokens.intersection(p_head_tokens):
                    # head argument is different
                    continue

                if not t_tail_tokens.intersection(p_tail_tokens):
                    # tail argument is different
                    continue

                # got a match
                match_found = True
                break

            # check match and remove relation from the buffer
            if match_found:
                matches += 1
                original_relations_buffer.remove(original_relation)

        # how we can check the intersection of these two sets and find the results
        re_stats.append(Stats(num_pred=len(predicted_relations), num_gold=len(original_relations), num_ok=matches))

    times = []
    stats_dir = result_dir / "stats"
    if stats_dir.exists():
        for stats_path in stats_dir.iterdir():
            with open(stats_path, "r") as stats_file:
                stats = json.load(stats_file)
            times.append(stats["duration_seconds"])

    return re_stats, times


# now we can show the results
print("==" * 20)
print("Final RE Scores:")

if __name__ == '__main__':
    def main():
        experiments = {
            "gpt-4o-ner": load_ner_metrics,
            "gpt-4o-re": load_re_metrics,
            "gpt-4o-end-to-end": load_re_metrics,

            "llama3_1_70b-md": load_ner_metrics,
            "llama3_1_70b-re": load_re_metrics,

            "llama3_2_1b-md": load_ner_metrics,
            "llama3_2_1b-re": load_re_metrics,

            "llama3_2_3b-md": load_ner_metrics,
            "llama3_2_3b-re": load_re_metrics,

            "llama3_3_70b-md": load_ner_metrics,
            "llama3_3_70b-re": load_re_metrics,
        }

        base_path = pathlib.Path(__file__).parent.parent.parent / "resources"
        results_dir = base_path / "results"
        pet_documents_path = base_path / "docs-small" / "pet" / "pet.jsonl"

        models = [
            "gpt-4o",
            "llama3_1_70b",
            "llama3_2_1b",
            "llama3_2_3b",
            "llama3_3_70b",
        ]

        for model in models:
            for targets in (results_dir / model).iterdir():
                if targets.name in ["re", "end-to-end", "isolated-re"]:
                    loader = load_re_metrics
                elif targets.name == "md":
                    loader = load_ner_metrics
                else:
                    raise AssertionError(f"Unknown target {targets.name}")

                stats, times = loader(targets, pet_documents_path)
                ps = [s.precision for s in stats]
                rs = [s.recall for s in stats]
                f1s = [s.f1 for s in stats]

                print("--" * 20)
                print(f"-- {model} ({targets.name})")
                print("--" * 20)
                print(f"Precision   : {np.mean(ps):5.1%} +- {np.std(ps):5.1%}")
                print(f"Recall      : {np.mean(rs):5.1%} +- {np.std(rs):5.1%}")
                print(f"F1          : {np.mean(f1s):5.1%} +- {np.std(f1s):5.1%}")
                if len(times) != 0:
                    print(f"Time (s)    : {np.mean(times):.1f} +- {np.std(times):.1f}")
                print("--" * 20)
                print()
        print("==" * 20)


    main()
