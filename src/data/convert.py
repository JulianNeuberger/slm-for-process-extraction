import pathlib
import typing
import random

import data
from data.conll03 import to_conll03
from data.pet import PetImporter, PetDocument, PetRelation, PetMention
from data.piqn import to_piqn, types_from_pet
from data.plmarker import to_plmarker
from data.unirel import to_unirel


def create_piqn_data(data_dir: pathlib.Path, train: typing.List[PetDocument], test: typing.List[PetDocument],
                     dev: typing.List[PetDocument]):
    data_dir.mkdir(exist_ok=True, parents=True)
    with open(data_dir / "train.json", "w") as f:
        f.write(to_piqn(train))
    with open(data_dir / "dev.json", "w") as f:
        f.write(to_piqn(dev))
    with open(data_dir / "test.json", "w") as f:
        f.write(to_piqn(test))
    docs = train + test + dev
    types = types_from_pet(docs)
    with open(data_dir / "types.json", "w") as f:
        f.write(types)


def create_ace_data(data_dir: pathlib.Path, train: typing.List[PetDocument], test: typing.List[PetDocument],
                    dev: typing.List[PetDocument]):
    data_dir.mkdir(exist_ok=True, parents=True)
    with open(data_dir / "train.txt", "w", encoding="utf8") as f:
        texts = [f"-DOCSTART-\t-DOCSTART-\tO\n\n{t}" for _, t in to_conll03(train)]
        f.write("\n\n".join(texts))
    with open(data_dir / "dev.txt", "w", encoding="utf8") as f:
        texts = [f"-DOCSTART-\t-DOCSTART-\tO\n\n{t}" for _, t in to_conll03(dev)]
        f.write("\n\n".join(texts))
    with open(data_dir / "test.txt", "w", encoding="utf8") as f:
        texts = [f"-DOCSTART-\t-DOCSTART-\tO\n\n{t}" for _, t in to_conll03(test)]
        f.write("\n\n".join(texts))


def create_unirel_data(data_dir: pathlib.Path, train: typing.List[PetDocument], test: typing.List[PetDocument],
                       dev: typing.List[PetDocument]):
    data_dir.mkdir(exist_ok=True, parents=True)
    with open(data_dir / "train_split.json", "w", encoding="utf8") as f:
        f.write(to_unirel(train))
    with open(data_dir / "valid_data.json", "w", encoding="utf8") as f:
        f.write(to_unirel(dev))
    with open(data_dir / "test_data.json", "w", encoding="utf8") as f:
        f.write(to_unirel(test))


def create_plmarker_data(data_dir: pathlib.Path, train: typing.List[PetDocument], test: typing.List[PetDocument],
                         dev: typing.List[PetDocument]):
    data_dir.mkdir(exist_ok=True, parents=True)
    with open(data_dir / "train.jsonl", "w", encoding="utf8") as f:
        f.write(to_plmarker(train))
    with open(data_dir / "dev.jsonl", "w", encoding="utf8") as f:
        f.write(to_plmarker(dev))
    with open(data_dir / "test.jsonl", "w", encoding="utf8") as f:
        f.write(to_plmarker(test))


def sanitize_doc(doc: PetDocument) -> None:
    # normalize document text
    doc.text = " ".join(t.text for t in doc.tokens)

    mentions_by_token_index = {}
    for m in doc.mentions:
        for i in m.token_document_indices:
            if i not in mentions_by_token_index:
                mentions_by_token_index[i] = []
            mentions_by_token_index[i].append(m)

    # fix common llm mistakes
    for i, r in enumerate(doc.relations):
        if r.type == "condition specification":
            doc.relations[i] = PetRelation(
                type="flow",
                head_mention_index=r.head_mention_index,
                tail_mention_index=r.tail_mention_index
            )
        if "_" in r.type:
            doc.relations[i] = PetRelation(
                type=r.type.replace("_", " "),
                head_mention_index=r.head_mention_index,
                tail_mention_index=r.tail_mention_index
            )

    for i, m in enumerate(doc.mentions):
        if "_" in m.type:
            doc.mentions[i] = PetMention(
                type=m.type.replace("_", " "),
                token_document_indices=m.token_document_indices,
            )

    # remove bad relations
    allowed_relations = {
        "activity": {
            "activity": {"flow"},
            "actor": {"actor performer", "actor recipient"},
            "xor gateway": {"flow"},
            "and gateway": {"flow"},
            "further specification": {"further specification"},
            "activity data": {"uses"}
        },
        "xor gateway": {
            "activity": {"flow"},
            "condition specification": {"flow"},
            "xor gateway": {"same gateway", "flow"},
            "and gateway": {"flow"},
        },
        "and gateway": {
            "activity": {"flow"},
            "and gateway": {"same gateway", "flow"},
            "xor gateway": {"flow"},
        },
        "condition specification": {
            "activity": {"flow"},
            "xor gateway": {"flow"},
            "and gateway": {"flow"},
        }
    }
    to_remove = []
    for i, r in enumerate(doc.relations):
        head_type = doc.mentions[r.head_mention_index].type
        tail_type = doc.mentions[r.tail_mention_index].type

        forward_allowed = (
                head_type in allowed_relations
                and tail_type in allowed_relations[head_type]
                and r.type in allowed_relations[head_type][tail_type]
        )

        if forward_allowed:
            continue

        backward_allowed = (
                tail_type in allowed_relations
                and head_type in allowed_relations[tail_type]
                and r.type in allowed_relations[tail_type][head_type]
        )

        head = doc.mentions[r.head_mention_index]
        tail = doc.mentions[r.tail_mention_index]

        if backward_allowed:
            print(f"Fixing {head.text(doc)} ({head_type}) -{r.type}-> {tail.text(doc)} ({tail_type}) by reversing")
            # reverse relation
            doc.relations[i] = PetRelation(
                type=r.type,
                head_mention_index=r.tail_mention_index,
                tail_mention_index=r.head_mention_index
            )
            continue

        print(f"Removing {head.text(doc)} ({head_type}) -{r.type}-> {tail.text(doc)} ({tail_type})")
        to_remove.append(r)

    for r in to_remove:
        doc.relations.remove(r)

    # remove invalid relations
    to_remove = []
    for r in doc.relations:
        if r.head_mention_index < 0 or r.head_mention_index >= len(doc.mentions):
            to_remove.append(r)
            continue
        if r.tail_mention_index < 0 or r.tail_mention_index >= len(doc.mentions):
            to_remove.append(r)
            continue
    for r in to_remove:
        doc.relations.remove(r)

    # sanitize overlapping mentions
    priority = [
        "condition specification",
        "xor gateway",
        "and gateway",
        "activity data",
        "activity",
        "actor",
        "further specification"
    ]
    to_remove = []
    for m in doc.mentions:
        if m.type not in priority:
            to_remove.append(m)
    for m in to_remove:
        doc.remove_mention(doc.mentions.index(m))

    for m_type in priority:
        mentions_to_remove: typing.Set[data.PetMention] = set()
        for mid, m in enumerate(doc.mentions):
            if m.type != m_type:
                continue
            for oid, o in enumerate(doc.mentions):
                if oid == mid:
                    continue
                overlapping_tokens = set(m.token_document_indices).intersection(set(o.token_document_indices))
                mentions_overlapping = len(overlapping_tokens) > 0
                if not mentions_overlapping:
                    continue
                if o.type == m_type:
                    # same priority, remove longer mention
                    if len(m.token_document_indices) < len(o.token_document_indices):
                        mentions_to_remove.add(o)
                        for rid, r in enumerate(doc.relations):
                            doc.relations[rid] = PetRelation(
                                type=r.type,
                                head_mention_index=r.head_mention_index if r.head_mention_index != oid else mid,
                                tail_mention_index=r.tail_mention_index if r.tail_mention_index != oid else mid,
                            )
                    else:
                        mentions_to_remove.add(m)
                        for rid, r in enumerate(doc.relations):
                            doc.relations[rid] = PetRelation(
                                type=r.type,
                                head_mention_index=r.head_mention_index if r.head_mention_index != mid else oid,
                                tail_mention_index=r.tail_mention_index if r.tail_mention_index != mid else oid,
                            )
                elif o.type == "xor gateway" and m.type == "condition specification":
                    # keep xor gate and remove its indices from the condition spec
                    print("previous: ==========================")
                    print(doc.mentions[oid].type, doc.mentions[oid].text(doc))
                    print(doc.mentions[mid].type, doc.mentions[mid].text(doc))
                    print("now: -------------------------------")

                    o_tokens = [tid for tid in o.token_document_indices if tid < m.token_document_indices[0]]
                    if len(o_tokens) == 0:
                        to_remove.append(o)
                        continue

                    doc.mentions[oid] = data.PetMention(
                        type=o.type,
                        token_document_indices=tuple(o_tokens),
                    )

                    print(doc.mentions[oid].type, doc.mentions[oid].text(doc))
                    print(doc.mentions[mid].type, doc.mentions[mid].text(doc))
                    print("====================================")
                else:
                    # lower priority
                    mentions_to_remove.add(o)
        for to_remove in mentions_to_remove:
            doc.remove_mention(doc.mentions.index(to_remove))

    # remove duplicate relations
    r_set = set()
    for r in doc.relations:
        r_tup = (r.head_mention_index, r.tail_mention_index, r.type)
        r_set.add(r_tup)
    doc.relations = [
        PetRelation(
            head_mention_index=r[0],
            tail_mention_index=r[1],
            type=r[2]
        )
        for r in r_set
    ]


def create_all_data(base_dir: pathlib.Path, subset: str, *, train: typing.List[PetDocument],
                    test: typing.List[PetDocument],
                    dev: typing.List[PetDocument]):
    base_dir.mkdir(exist_ok=True, parents=True)
    approaches = {
        "plmarker": create_plmarker_data,
        "unirel": create_unirel_data,
        "piqn": create_piqn_data,
        "ace": create_ace_data,
    }

    for n, f in approaches.items():
        data_dir = base_dir / n / subset
        f(data_dir, train, test, dev)


def collect_synth_data(data_dir: pathlib.Path) -> typing.List[PetDocument]:
    all_docs: typing.List[PetDocument] = []
    for f_path in data_dir.iterdir():
        docs = PetImporter(f_path).do_import()
        all_docs.extend(docs)
    return all_docs


def build_splits(
        docs: typing.List[PetDocument],
        *,
        seed: int,
        dev: float = 0.15,
        test: float = 0.2,
) -> typing.Tuple[typing.List[PetDocument], typing.List[PetDocument], typing.List[PetDocument]]:
    indices = list(range(len(docs)))
    random.seed(seed)
    random.shuffle(indices)

    train_indices = indices[:int(len(indices) * (1 - test))]
    test_indices = indices[int(len(indices) * (1 - test)):]
    dev_indices = train_indices[int(len(train_indices) * (1 - dev)):]
    train_indices = train_indices[:int(len(train_indices) * (1 - dev))]

    assert len(docs) == len(train_indices) + len(dev_indices) + len(test_indices)
    assert len(set(train_indices).intersection(set(dev_indices))) == 0
    assert len(set(test_indices).intersection(set(dev_indices))) == 0
    assert len(set(test_indices).intersection(set(train_indices))) == 0
    assert len(set(indices)) == len(set(train_indices) | set(dev_indices) | set(test_indices))

    return (
        [docs[i] for i in train_indices],
        [docs[i] for i in dev_indices],
        [docs[i] for i in test_indices],
    )


if __name__ == "__main__":
    def main():
        dev_ratio = 0.15
        test_ratio = 0.2
        seeds = [42, 43, 44, 45, 46]
        resources_dir = pathlib.Path(__file__).parent.parent.parent / "resources"
        out_dir = resources_dir / "processed-small"
        docs_dir = "docs-small"

        for seed in seeds:
            print("Only PET data")
            docs = PetImporter(resources_dir / docs_dir / "pet" / "pet.jsonl").do_import()
            for d in docs:
                sanitize_doc(d)
            pet_train, pet_dev, pet_test = build_splits(docs, dev=dev_ratio, test=test_ratio, seed=seed)
            create_all_data(out_dir, subset=f"pet/{seed}", train=pet_train, test=pet_test, dev=pet_dev)

            if (resources_dir / docs_dir / "sbvr").exists():
                print("Only SBVR hint data")
                docs = collect_synth_data(resources_dir / docs_dir / "sbvr")
                for d in docs:
                    sanitize_doc(d)
                sbvr_train, sbvr_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
                create_all_data(out_dir, subset=f"sbvr/{seed}", train=sbvr_train, test=pet_test, dev=sbvr_dev)

            if (resources_dir / docs_dir / "no_hints").exists():
                print("Only no hint data")
                docs = collect_synth_data(resources_dir / docs_dir / "no_hints")
                for d in docs:
                    sanitize_doc(d)
                no_hints_train, no_hints_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
                create_all_data(out_dir, subset=f"no_hints/{seed}", train=no_hints_train, test=pet_test, dev=no_hints_dev)

            if (resources_dir / docs_dir / "image").exists():
                print("Only image hint data")
                docs = collect_synth_data(resources_dir / docs_dir / "image")
                for d in docs:
                    sanitize_doc(d)
                image_train, image_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
                create_all_data(out_dir, subset=f"image/{seed}", train=image_train, test=pet_test, dev=image_dev)

            if (resources_dir / docs_dir / "combined").exists():
                print("Combined hint data")
                docs = collect_synth_data(resources_dir / docs_dir / "combined")
                for d in docs:
                    sanitize_doc(d)
                comb_train, comb_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
                create_all_data(out_dir, subset=f"combined/{seed}", train=comb_train, test=pet_test, dev=comb_dev)


    main()
