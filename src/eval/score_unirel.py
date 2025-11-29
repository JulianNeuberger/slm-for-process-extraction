import json
import pathlib
import typing


def get_f1_scores(data_base_path: pathlib.Path, experiment: str) -> typing.List[float]:
    f1s = []
    experiment_path = data_base_path / "results" / "unirel" / experiment
    for seed in experiment_path.iterdir():
        if not seed.is_dir():
            continue

        for fold in seed.iterdir():
            if not fold.is_dir():
                continue

            num_gold = 0
            num_pred = 0
            num_ok = 0

            result_file =  fold / "test_predict_sard.json"
            if not result_file.exists():
                continue
            with open(result_file, "r") as f:
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
            f1 = 2*p*r / (p + r)
            f1s.append(f1)
    return f1s