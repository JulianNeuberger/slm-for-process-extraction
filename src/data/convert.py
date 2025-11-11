import pathlib
import typing
import random

import data
from data.conll03 import to_conll03
from data.pet import PetImporter, PetDocument, PetRelation
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

    # fix common llm mistakes
    for i, r in enumerate(doc.relations):
        if r.type == "condition specification":
            doc.relations[i] = PetRelation(
                type="flow",
                head_mention_index=r.head_mention_index,
                tail_mention_index=r.tail_mention_index
            )

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

    to_remove = []
    for i, r in enumerate(doc.relations):
        if "_" in r.type:
            r = PetRelation(
                type=r.type.replace("_", " "),
                head_mention_index=r.head_mention_index,
                tail_mention_index=r.tail_mention_index,
            )
            doc.relations[i] = r
        if r.type not in [
            "flow", "uses", "actor performer", "actor recipient", "further specification", "same gateway"
        ]:
            to_remove.append(r)
    for r in to_remove:
        doc.relations.remove(r)

    for m_type in priority:
        mentions_to_remove: typing.Set[data.PetMention] = set()
        for mid, m in enumerate(doc.mentions):
            if m.type != m_type:
                continue
            for oid, o in enumerate(doc.mentions):
                if oid == mid:
                    continue
                mentions_overlapping = len(
                    set(m.token_document_indices).intersection(set(o.token_document_indices))) > 0
                if not mentions_overlapping:
                    continue
                if o.type == m_type:
                    # same priority, remove longer mention
                    if len(m.token_document_indices) < len(o.token_document_indices):
                        mentions_to_remove.add(o)
                    else:
                        mentions_to_remove.add(m)
                else:
                    # lower priority
                    mentions_to_remove.add(o)
        for to_remove in mentions_to_remove:
            doc.remove_mention(doc.mentions.index(to_remove))


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

        for seed in seeds:
            print("Only PET data")
            pet_only_out_dir = resources_dir / "processed" / "pet"
            docs = PetImporter(resources_dir / "docs" / "pet" / "pet.jsonl").do_import()
            for d in docs:
                sanitize_doc(d)
            pet_train, pet_dev, pet_test = build_splits(docs, dev=dev_ratio, test=test_ratio, seed=seed)
            create_all_data(pet_only_out_dir, subset=f"{seed}", train=pet_train, test=pet_test, dev=pet_dev)

            print("Only SBVR hint data")
            sbvr_only_out_dir = resources_dir / "processed" / "sbvr"
            docs = collect_synth_data(resources_dir / "docs" / "sbvr")
            for d in docs:
                sanitize_doc(d)
            sbvr_train, sbvr_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
            create_all_data(sbvr_only_out_dir, subset=f"{seed}", train=sbvr_train, test=pet_test, dev=sbvr_dev)

            print("Only image hint data")
            combined_out_dir = resources_dir / "processed" / "image"
            docs = collect_synth_data(resources_dir / "docs" / "image")
            for d in docs:
                sanitize_doc(d)
            image_train, image_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
            create_all_data(combined_out_dir, subset=f"{seed}", train=image_train, test=pet_test, dev=image_dev)

            print("Combined hint data")
            combined_out_dir = resources_dir / "processed" / "combined"
            docs = collect_synth_data(resources_dir / "docs" / "combined")
            for d in docs:
                sanitize_doc(d)
            comb_train, comb_dev, _ = build_splits(docs, dev=dev_ratio, test=0, seed=seed)
            create_all_data(combined_out_dir, subset=f"{seed}", train=comb_train, test=pet_test, dev=comb_dev)


    main()
