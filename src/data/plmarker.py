import json
import typing

from data.pet import PetDocument


def to_plmarker(dataset: typing.List[PetDocument]) -> str:
    docs = [doc_to_plmarker(d) for d in dataset]
    re_labels = set()
    ner_labels = set()
    for d in docs:
        for s in d["ner"]:
            for m in s:
                ner_labels.add(m[2])
        for s in d["relations"]:
            for r in s:
                re_labels.add(r[4])
    return "\n".join(json.dumps(d) for d in docs)


def doc_to_plmarker(doc: PetDocument) -> typing.Dict:
    sentences = [[t.text for t in doc.tokens]]
    mentions_by_sentence = [[]]
    relations_by_sentence = [[]]

    mentions_by_id = {}
    for m_id, m in enumerate(doc.mentions):
        mention = [m.token_document_indices[0], m.token_document_indices[-1], m.type.replace(" ", "_")]
        mentions_by_id[m_id] = mention
        mentions_by_sentence[0].append(mention)
        converted_text = " ".join(t.text for t in doc.tokens[mention[0]: mention[1] + 1])
        original_text = doc.mentions[m_id].text(doc)
        assert converted_text == original_text, f"Expected '{original_text}' but got '{converted_text}'"

    for r in doc.relations:
        head_mention = doc.mentions[r.head_mention_index]
        tail_mention = doc.mentions[r.tail_mention_index]
        relation = mentions_by_id[r.head_mention_index][0:2] + mentions_by_id[r.tail_mention_index][0:2] + [r.type.replace(" ", "_")]
        assert not any(relation == existing_r for existing_r in relations_by_sentence[0]), f"{relation} ({doc.name}) ({relations_by_sentence[0]}), {doc.tokens[relation[0]: relation[1] + 1]} -> {doc.tokens[relation[2]: relation[3] + 1]}"
        relations_by_sentence[0].append(relation)
        assert " ".join(t.text for t in doc.tokens[relation[0]: relation[1] + 1]) == head_mention.text(doc)
        assert " ".join(t.text for t in doc.tokens[relation[2]: relation[3] + 1]) == tail_mention.text(doc)
        assert relation[4] == r.type.replace(" ", "_")

    return {
        "doc_key": doc.id,
        "sentences": sentences,
        "ner": mentions_by_sentence,
        "relations": relations_by_sentence
    }
