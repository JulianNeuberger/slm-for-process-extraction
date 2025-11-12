import pathlib
import traceback
import typing

import dotenv

import data
import prompts
from annotate import base

dotenv.load_dotenv()


class MentionParser(base.BaseParser):
    @staticmethod
    def parse_line(
            line: str, document: data.PetDocument
    ) -> typing.List[data.PetMention]:
        split_line = line.split("\t")
        split_line = tuple(e for e in split_line if e.strip() != "")

        if len(split_line) < 3 or len(split_line) > 4:
            print("Malformed line:", line)
            raise ValueError(
                f"Skipping line {split_line}, as it is not formatted "
                f"properly, expected between 3 and 4 arguments."
            )

        if len(split_line) == 3:
            mention_text, mention_type, sentence_id = split_line
        else:
            mention_text, mention_type, sentence_id, explanation = split_line
            # print(f"Explanation for {mention_text}: {explanation}")

        try:
            sentence_id = int(sentence_id)
        except ValueError:
            print(f"No numerical sentence id in line: {line}")
            raise ValueError(f"Invalid sentence index '{sentence_id}', skipping line.")

        sentence = document.sentences[sentence_id]

        mention_text = mention_text.lower()
        mention_tokens = mention_text.split(" ")

        res = []
        found = False
        for i, token in enumerate(sentence):
            candidates = sentence[i: i + len(mention_tokens)]
            candidate_text = " ".join(c.text.lower() for c in candidates)

            if candidate_text.lower() != mention_text.lower():
                continue

            res.append(
                data.PetMention(
                    token_document_indices=tuple(
                        c.index_in_document for c in candidates
                    ),
                    type=mention_type.lower().strip(),
                )
            )
            found = True
        if not found:
            print(f"Did not find predicted mention '{mention_text}'.")
        return res

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        parsed_mentions: typing.List[data.PetMention] = []
        num_parse_errors = 0
        for line in string.splitlines(keepends=False):
            line = line.strip()
            if line == "":
                continue
            if "\t" not in line:
                continue

            try:
                mentions_from_line = self.parse_line(line, document)
                parsed_mentions.extend(mentions_from_line)
            except Exception:
                num_parse_errors += 1
                print("Error during parsing of line, skipping line. Error was:")
                print(traceback.format_exc())

        doc = data.PetDocument(
            id=document.id,
            text=document.text,
            name=document.name,
            category=document.category,
            tokens=[t.copy() for t in document.tokens],
            mentions=parsed_mentions,
            relations=[],
            entities=[],
        )
        return doc


class LLMMentionAnnotator(base.BaseAnnotator):
    def get_parser(self) -> base.BaseParser:
        return MentionParser()

    def get_prompt_template(self) -> prompts.Prompt:
        prompt_folder = pathlib.Path(__file__).parent.parent.parent.resolve() / "resources" / "prompts"
        return prompts.Prompt(prompt_folder / "annotate-mentions.txt")

    def _format(self, document: data.PetDocument) -> str:
        return "\n".join(f"{i}: {' '.join(t.text for t in s)}" for i, s in enumerate(document.sentences))

    def get_text_formatter(self) -> typing.Callable[[data.PetDocument], str]:
        return self._format
