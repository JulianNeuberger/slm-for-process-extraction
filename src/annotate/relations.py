import pathlib
import typing

import dotenv

import data
import prompts
from annotate import util, base

dotenv.load_dotenv()


class RelationParser(base.BaseParser):
    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=["relations"])
        total_errors = 0
        for line in string.splitlines(keepends=False):
            if "\t" not in line:
                print(f"Skipping non-tab-separated line {line}.")
                continue
            split_line = line.split("\t")
            if len(split_line) == 3 or len(split_line) > 4:
                relation_type, head_index, tail_index = split_line
            elif len(split_line) == 4:
                relation_type, head_index, tail_index, explanation = split_line
            else:
                print(
                    f"Expected exactly 3-4 arguments in line {line}, got {len(split_line)}. Skipping."
                )
                total_errors += 1
                continue
            relation_type = relation_type.lower().strip()
            try:
                head_index = int(head_index)
                tail_index = int(tail_index)
            except ValueError:
                total_errors += 1
                continue
            if head_index >= len(document.mentions):
                continue
            if tail_index >= len(document.mentions):
                continue
            document.relations.append(
                data.PetRelation(
                    type=relation_type,
                    head_mention_index=head_index,
                    tail_mention_index=tail_index,
                )
            )
        return document


class LLMRelationsAnnotator(base.BaseAnnotator):
    def get_parser(self) -> base.BaseParser:
        return RelationParser()

    def get_text_formatter(self) -> typing.Callable[[data.PetDocument], str]:
        return util.format_document_text_with_entity_mentions

    def get_prompt_template(self) -> prompts.Prompt:
        prompt_folder = pathlib.Path(__file__).parent.parent.parent.resolve() / "resources" / "prompts"
        return prompts.Prompt(prompt_folder / "annotate-relations.txt")
