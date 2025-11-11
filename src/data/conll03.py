import pathlib
import random
import re
import typing

from data.pet import PetImporter, PetDocument, PetToken


def to_conll03(dataset: typing.List[PetDocument]) -> typing.List[typing.Tuple[str, str]]:
    return [(str(d.id), doc_to_conll(d)) for d in dataset]


def doc_to_conll(doc: PetDocument) -> str:
    tags = {}
    for t in doc.tokens:
        tags[t] = "O"
    for m in doc.mentions:
        tag = re.sub(r"\s+", "_", m.type)
        for i, index in enumerate(m.token_document_indices):
            if i == 0:
                tags[doc.tokens[index]] = f"B-{tag}"
            else:
                tags[doc.tokens[index]] = f"I-{tag}"

    sentences = []
    for sentence in doc.sentences:
        sentences.append(sentence_to_conll(sentence, tags))
    return "\n\n".join(sentences)


def sentence_to_conll(sentence: typing.List[PetToken], tags: typing.Dict[PetToken, str]) -> str:
    lines = []
    for t in sentence:
        line = f"{t.text}\t{t.pos_tag}\t{tags[t]}"
        lines.append(line)
    return "\n".join(lines)


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

        (data_dir / "ace").mkdir(exist_ok=True, parents=True)

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
