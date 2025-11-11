import json
import typing

from data.pet import PetDocument


def to_unirel(dataset: typing.List[PetDocument]):
    as_unirel = [document_to_unirel(d) for d in dataset]
    return json.dumps(as_unirel)


def document_to_unirel(doc: PetDocument) -> typing.Dict:
    as_json = {
        "text": doc.text,
        "id": doc.id,
        "relation_list": [],
        "entity_list": []
    }

    token_starts = {}
    index = 0
    for t in doc.tokens:
        token_starts[t.index_in_document] = index
        index += len(t.text) + 1

    entities = {}
    for i, m in enumerate(doc.mentions):
        start_char = token_starts[m.token_document_indices[0]]
        end_char = token_starts[m.token_document_indices[-1]] + len(doc.tokens[m.token_document_indices[-1]].text)
        entity = {
            "text": m.text(doc),
            "type": m.type,
            "tok_span": [m.token_document_indices[0], m.token_document_indices[-1] + 1],
            "char_span": [start_char, end_char],
        }
        doc_text = " ".join(t.text for t in doc.tokens)
        assert doc.text == doc_text, f"Expected '{doc.text}' but got '{doc_text}'"
        token_text = " ".join(t.text for t in doc.tokens[entity["tok_span"][0]: entity["tok_span"][1]])
        assert token_text == entity["text"], f"Expected '{entity['text']}', got '{token_text}'"
        char_text = doc.text[entity["char_span"][0]: entity["char_span"][1]]
        assert char_text == entity["text"], f"Expected '{entity['text']}', got '{char_text}'"
        as_json["entity_list"].append(entity)
        entities[i] = entity

    for r in doc.relations:
        sub = entities[r.head_mention_index]
        obj = entities[r.tail_mention_index]
        relation = {
            "subject": sub["text"],
            "object": obj["text"],
            "predicate": r.type,
            "subj_char_span": sub["char_span"],
            "obj_char_span": obj["char_span"],
            "subj_tok_span": sub["tok_span"],
            "obj_tok_span": obj["tok_span"],
        }
        as_json["relation_list"].append(relation)

    return as_json
