import csv
import json
import pathlib
import re
import typing

import pandas as pd
import tabulate
from scipy import stats

from eval.scoring import Scores, ScoresAccumulator


def import_experiment(
    base_dir: pathlib.Path,
    loader_fn: typing.Callable[[pathlib.Path], Scores]
) -> typing.Dict[str, ScoresAccumulator]:
    ret: typing.Dict[str, ScoresAccumulator] = {}
    for seed in base_dir.iterdir():
        if not seed.is_dir():
            print(f"Skipping file {seed.name}.")
            continue

        if seed.name not in ret:
            ret[seed.name] = ScoresAccumulator()
        ret[seed.name] += loader_fn(seed)
    return ret


def import_piqn(directory: pathlib.Path) -> Scores:
    model = next(directory.iterdir())
    run = next(model.iterdir())

    results_path = run / "eval_test.csv"

    with open(results_path, "r") as results_file:
        reader = csv.DictReader(results_file, delimiter=";")
        values = next(reader)
        return Scores(
            p=float(values['ner_prec_micro']) / 100.0,
            r=float(values['ner_rec_micro']) / 100.0,
            f1=float(values['ner_f1_micro']) / 100.0
        )


def import_uni_rel(directory: pathlib.Path) -> Scores:
    results_path = directory / "test_predict_sard.json"
    num_gold = 0
    num_pred = 0
    num_ok = 0

    with open(results_path, "r") as f:
        docs = json.load(f)

    for d in docs:
        true = d["gold_spo_list"]
        pred = d["pred_spo_list"]

        true_rels = [(r[0].split(" "), r[1], r[2].split(" ")) for r in true]
        pred_rels = [(r[0].split(" "), r[1], r[2].split(" ")) for r in pred]

        num_gold += len(true_rels)
        num_pred += len(pred_rels)

        for t_head, t_type, t_tail in true_rels:
            for i, (p_head, p_type, p_tail) in enumerate(pred_rels):
                if t_type != p_type:
                    continue
                if not set(t_head).intersection(set(p_head)):
                    continue
                if not set(t_tail).intersection(set(p_tail)):
                    continue
                num_ok += 1
                pred_rels.pop(i)
                break
    p = num_ok / num_pred
    r = num_ok / num_gold
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    return Scores(p=p, r=r, f1=f1)





def import_ace(directory: pathlib.Path) -> Scores:
    results_path = directory / "results.log"

    with open(results_path, "r") as results_file:
        log = results_file.read()
        raw_scores = re.findall(r"MICRO_AVG: acc \d\.\d+ - f1-score (\d.\d+)", log)
        return Scores(p=0, r=0, f1=float(raw_scores[-1]))


def import_plmarker(directory: pathlib.Path) -> Scores:
    results_path = directory / "results.json"

    with open(results_path, "r") as results_file:
        values = json.load(results_file)
        return Scores(p=0, r=0, f1=values['f1_with_ner_'])


def import_relative_experiments(
    base_dir: pathlib.Path,
    experiments: typing.List[str],
    baseline: str, import_fn: typing.Callable[[pathlib.Path], Scores]
) -> pd.DataFrame:
    assert baseline in experiments

    run_scores = {}
    for e in experiments:
        run_scores[e] = ScoresAccumulator()
        scores = import_experiment(base_dir / e, import_fn)
        for seed, seed_scores in scores.items():
            run_scores[e] += seed_scores

    data = {
        "experiment": [],
        "f1": [],
        "f1_std": []
    }

    for experiment, values in run_scores.items():
        scores = values.to_scores()
        data["experiment"].append(experiment)
        data["f1"].append(scores.f1 * 100)
        data["f1_std"].append(scores.f1_std * 100)

    df = pd.DataFrame(data)
    return df


def import_subset_experiment(base_dir: pathlib.Path, import_fn: typing.Callable[[pathlib.Path], Scores]) -> pd.DataFrame:
    subset_scores = import_experiment(base_dir / "subset", import_fn)
    # subset_scores = {
    #     s: sum(v.values(), ScoresAccumulator()) for s, v in subset_scores.items()
    # }
    data = {
        "amount": [],
        "score": []
    }
    for amount, scores in subset_scores.items():
        for s in scores.f1:
            data["amount"].append(float(amount) / 10)
            data["score"].append(s * 100.0)
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    def main():
        experiments = ["pet-cv", "synth-cv", "mixed-cv", "fine-tune"]
        baseline = "pet-cv"

        # df = import_relative_experiments(pathlib.Path("../../data/results/plmarker"),  experiments, baseline, import_plmarker)
        # print(tabulate.tabulate(df, headers="keys", tablefmt="psql"))
        # subset_df = import_subset_experiment(pathlib.Path("../../data/results/plmarker"), import_plmarker)


        # print("PIQN (abs)")
        # import_experiment()
        # print_table(get_absolute_scores(import_piqn(pathlib.Path("../../data/results/piqn/eval"))))
        # print("PIQN (rel)")
        # print_table(get_relative_scores(import_piqn(pathlib.Path("../../data/results/piqn/eval"))))
        #
        # print()
        # print()
        #
        # print("ACE (abs)")
        # print_table(get_absolute_scores(import_ace(pathlib.Path("../../data/results/ace"))))
        # print("ACE (rel)")
        # print_table(get_relative_scores(import_ace(pathlib.Path("../../data/results/ace"))))
        #
        # print()
        # print()
        #
        # print("UNIREL (abs)")
        # print_table(get_absolute_scores(import_uni_rel(pathlib.Path("../../data/results/unirel"))))
        # print("UNIREL (rel)")
        # print_table(get_relative_scores(import_uni_rel(pathlib.Path("../../data/results/unirel"))))
        #
        # print()
        # print()
        #
        # print("PLMARKER (abs)")
        # print_table(get_absolute_scores(import_plmarker(pathlib.Path("../../data/results/plmarker"))))
        # print("PLMARKER (rel)")
        # print_table(get_relative_scores(import_plmarker(pathlib.Path("../../data/results/plmarker"))))

    main()
