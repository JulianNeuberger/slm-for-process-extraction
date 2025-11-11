import json
import pathlib
import random
import typing

from data.pet import PetDocument, PetImporter


def to_piqn(pet_data: typing.List[PetDocument]) -> str:
    dataset = []
    for d in pet_data:
        dataset.extend(pet_document_to_piqn(d))
    return json.dumps(dataset)


def pet_document_to_piqn(pet_document: PetDocument) -> typing.List[typing.Dict]:
    mentions_by_start = {}
    for m in pet_document.mentions:
        mentions_by_start[m.token_document_indices[0]] = m

    converted_sentences = []
    sentences = pet_document.sentences
    for i, s in enumerate(pet_document.sentences):
        l_tokens = [t.text for t in sentences[i - 1]] if i > 0 else []
        r_tokens = [t.text for t in sentences[i + 1]] if i < len(pet_document.sentences) - 1 else []
        tokens = [t.text for t in s]
        entities = []
        for t in s:
            if t.index_in_document in mentions_by_start:
                m = mentions_by_start[t.index_in_document]
                start = s.index(t)
                end = start + len(m.token_document_indices)
                entities.append({
                    "type": m.type,
                    "start": start,
                    "end": end,
                })
        converted_sentences.append({
            "tokens": tokens,
            "entities": entities,
            "relations": [],
            "ltokens": l_tokens,
            "rtokens": r_tokens,
        })

    return converted_sentences


def types_from_pet(pet_data: typing.List[PetDocument]) -> str:
    ret = {
        "entities": {},
        "relations": {}
    }
    for d in pet_data:
        for m in d.mentions:
            if m.type not in ret["entities"]:
                ret["entities"][m.type] = {
                    "verbose": m.type,
                    "short": m.type
                }
    return json.dumps(ret)


if __name__ == "__main__":
    def main():
        dev_split = 0.1
        test_split = 0.2

        data_dir = pathlib.Path(__file__).parent.parent.parent / "data"
        docs_path = data_dir / "pet.jsonl"
        docs = PetImporter(docs_path).do_import()
        random.shuffle(docs)

        last_train_index = int(len(docs) * (1 - test_split - dev_split))
        last_dev_index = int(len(docs) * (1 - test_split))
        train = docs[:last_train_index]
        dev = docs[last_train_index:last_dev_index]
        test = docs[last_dev_index:]

        (data_dir / "piqn").mkdir(exist_ok=True, parents=True)

        with open(data_dir / "piqn" / "train.json", "w") as f:
            f.write(to_piqn(train))
        with open(data_dir / "piqn" / "dev.json", "w") as f:
            f.write(to_piqn(dev))
        with open(data_dir / "piqn" / "test.json", "w") as f:
            f.write(to_piqn(test))
        types = types_from_pet(docs)
        with open(data_dir / "piqn" / "types.json", "w") as f:
            f.write(types)


    main()
