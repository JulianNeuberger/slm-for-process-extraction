import pathlib
import typing

import dotenv

import data
import prompts
from annotate import util, base
from annotate.base import BaseParser

dotenv.load_dotenv()


class EntityParser(base.BaseParser):
    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=["entities"])
        for line in string.splitlines(keepends=False):
            if " " not in line:
                try:
                    mention_id = int(line)
                    if mention_id >= len(document.mentions):
                        continue
                    mention_ids = []
                except ValueError:
                    print(f"Skipping non space-separated line '{line}'!")
                    continue
            else:
                mention_ids = []
                for i in line.split(" "):
                    try:
                        mention_id = int(i)
                        if mention_id >= len(document.mentions):
                            continue
                        mention_ids.append(mention_id)
                    except ValueError:
                        pass
            mentions = []
            for i in mention_ids:
                if i >= len(document.mentions):
                    continue
                mentions.append(document.mentions[i])
            mention_types = set(m.type for m in mentions)
            if len(mention_types) > 1:
                print(f"Extracted multi-type entity, with mentions {mentions}.")
            document.entities.append(data.PetEntity(mention_indices=tuple(mention_ids)))
        for i, mention in enumerate(document.mentions):
            if any([i in e.mention_indices for e in document.entities]):
                continue
            document.entities.append(data.PetEntity(mention_indices=(i,)))
        return document


class LLMEntitiesAnnotator(base.BaseAnnotator):
    def get_prompt_template(self) -> prompts.Prompt:
        prompt_dir = pathlib.Path(__file__).parent.parent.parent / "resources" / "prompts"
        return prompts.Prompt(prompt_dir / "annotate-entities.txt")

    def _format_text(self, document: data.PetDocument) -> str:
        return util.format_document_text_with_entity_mentions(
            document=document,
            only_types=["Activity Data", "Actor"]
        )

    def get_text_formatter(self) -> typing.Callable[[data.PetDocument], str]:
        return self._format_text

    def get_parser(self) -> BaseParser:
        return EntityParser()
