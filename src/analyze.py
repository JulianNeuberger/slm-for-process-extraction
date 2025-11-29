import json
import pathlib
import typing

import data


def collect_synth_data(data_dir: pathlib.Path) -> typing.List[data.PetDocument]:
    all_docs: typing.List[data.PetDocument] = []
    for f_path in data_dir.iterdir():
        docs = data.PetImporter(f_path).do_import()
        all_docs.extend(docs)
    return all_docs


def collect_violations(data_dir: pathlib.Path) -> typing.Dict[str, int]:
    all_violations = {}
    for f_path in data_dir.iterdir():
        if f_path.suffix == ".json":
            with open(f_path) as f:
                violations = json.load(f)
                for v, k in violations.items():
                    if v not in all_violations:
                        all_violations[v] = 0
                    all_violations[v] += k
    return all_violations



resources_dir = pathlib.Path(__file__).parent.parent / "resources"
docs = collect_synth_data(resources_dir / "docs" / "sbvr")

matrix = {}
for doc in docs:
    for r in doc.relations:
        head = doc.mentions[r.head_mention_index]
        tail = doc.mentions[r.tail_mention_index]
        if head.type not in matrix:
            matrix[head.type] = {}
        if tail.type not in matrix[head.type]:
            matrix[head.type][tail.type] = {}
        if r.type not in matrix[head.type][tail.type]:
            matrix[head.type][tail.type][r.type] = 0
        matrix[head.type][tail.type][r.type] += 1

max_h_len = max(len(t) for t in matrix)
max_t_len = max(len(t) for h in matrix for t in matrix[h])
for head_type in matrix:
    for tail_type in matrix[head_type]:
        print(f"{head_type:>{max_h_len}} -> {tail_type:<{max_t_len}}", matrix[head_type][tail_type])

print(f"There are {len(docs)} documents in total.")

violations = collect_violations(resources_dir / "models" / "selected")
longest_key = max(len(k) for k in violations.keys())
for v_name, v_value in sorted(violations.items(), key=lambda x: x[1], reverse=True):
    print(f"{v_name:<{longest_key}}: {v_value}")
print(f"Total of {sum(v for v in violations.values())} violations.")