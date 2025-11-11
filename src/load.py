import csv
import dataclasses
import json
import pathlib
import typing


@dataclasses.dataclass
class ModelInfo:
    id: str
    name: str
    json_string: str
    namespace: str

    @property
    def model_json(self) -> typing.Dict[str, typing.Any]:
        return json.loads(self.json_string)

    @property
    def row(self):
        return {
            "Model ID": self.id,
            "Name": self.name,
            "Namespace": self.namespace,
            "Model JSON": self.json_string,
        }


@dataclasses.dataclass
class SBVR:
    rules: typing.List[str]
    vocab: typing.List[str]

    @property
    def row(self):
        return {
            "SBVR Rules": "\n".join(self.rules),
            "SBVR Vocabulary": "\n".join(self.vocab),
        }


@dataclasses.dataclass
class LLMCompletion:
    text: str
    prompt_tokens: int
    completion_tokens: int

    def as_row(self, name: str):
        return {
            f"{name}": self.text,
            f"{name} Prompt Tokens": self.prompt_tokens,
            f"{name} Completion Tokens": self.completion_tokens,
        }


@dataclasses.dataclass
class ProcessDescriptions:
    from_sbvr: LLMCompletion
    from_picture: LLMCompletion
    from_both: LLMCompletion

    @property
    def row(self):
        return {
            **self.from_sbvr.as_row("Description (SBVR)"),
            **self.from_picture.as_row("Description (Picture)"),
            **self.from_both.as_row("Description (Combined)")
        }


@dataclasses.dataclass
class Annotations:
    no_hints: LLMCompletion
    sbvr_hints: LLMCompletion
    picture_hints: LLMCompletion
    combined_hints: LLMCompletion

    @property
    def row(self):
        return {
            **self.no_hints.as_row("Annotations (No Hints)"),
            **self.sbvr_hints.as_row("Annotations (SBVR Hints)"),
            **self.picture_hints.as_row("Annotations (Picture Hints)"),
            **self.combined_hints.as_row("Annotations (Combined Hints)")
        }


@dataclasses.dataclass
class ModelSBVR:
    model: ModelInfo
    sbvr: SBVR

    @property
    def row(self):
        return {
            **self.model.row,
            **self.sbvr.row,
        }


@dataclasses.dataclass
class DescribedModel(ModelSBVR):
    descriptions: ProcessDescriptions

    @property
    def row(self):
        return {
            **super().row,
            **self.descriptions.row,
        }


@dataclasses.dataclass
class AnnotatedModel(DescribedModel):
    annotations: Annotations

    @property
    def row(self):
        return {
            **super().row,
            **self.annotations.row,
        }


def load_raw_models(models_path: pathlib.Path) -> typing.Generator[ModelInfo, None, None]:
    with (open(models_path, "r", encoding="utf-8") as f):
        csv_reader = csv.reader(f, delimiter=",")
        header = next(csv_reader)

        model_id_index = header.index("Model ID")
        model_name_index = header.index("Name")
        model_json_index = header.index("Model JSON")
        model_namespace_index = header.index("Namespace")

        for row in csv_reader:
            yield ModelInfo(
                id=row[model_id_index],
                name=row[model_name_index],
                json_string=row[model_json_index],
                namespace=row[model_namespace_index],
            )


def load_sbvr_models(models_path: pathlib.Path) -> typing.Generator[ModelSBVR, None, None]:
    with (open(models_path, "r", encoding="utf-8") as f):
        csv_reader = csv.reader(f, delimiter=",")
        header = next(csv_reader)

        model_id_index = header.index("Model ID")
        model_name_index = header.index("Name")
        model_json_index = header.index("Model JSON")
        model_namespace_index = header.index("Namespace")

        sbvr_rules_index = header.index("SBVR Rules")
        sbvr_vocab_index = header.index("SBVR Vocabulary")

        for row in csv_reader:
            model = ModelInfo(
                id=row[model_id_index],
                name=row[model_name_index],
                json_string=row[model_json_index],
                namespace=row[model_namespace_index],
            )

            yield ModelSBVR(
                model=model,
                sbvr=SBVR(
                    rules=row[sbvr_rules_index].split("\n"),
                    vocab=row[sbvr_vocab_index].split("\n")
                )
            )


def load_described_models(models_path: pathlib.Path) -> typing.Generator[DescribedModel, None, None]:
    with (open(models_path, "r", encoding="utf-8") as f):
        csv_reader = csv.reader(f, delimiter=",")
        header = next(csv_reader)

        model_id_index = header.index("Model ID")
        model_name_index = header.index("Name")
        model_json_index = header.index("Model JSON")
        model_namespace_index = header.index("Namespace")

        sbvr_rules_index = header.index("SBVR Rules")
        sbvr_vocab_index = header.index("SBVR Vocabulary")

        from_sbvr_text_index = header.index("Description (SBVR)")
        from_sbvr_prompt_tokens_index = header.index("Description (SBVR) Prompt Tokens")
        from_sbvr_completion_tokens_index = header.index("Description (SBVR) Completion Tokens")

        from_picture_text_index = header.index("Description (Picture)")
        from_picture_prompt_tokens_index = header.index("Description (Picture) Prompt Tokens")
        from_picture_completion_tokens_index = header.index("Description (Picture) Completion Tokens")

        from_both_text_index = header.index("Description (Combined)")
        from_both_prompt_tokens_index = header.index("Description (Combined) Prompt Tokens")
        from_both_completion_tokens_index = header.index("Description (Combined) Completion Tokens")



        for row in csv_reader:
            model = ModelInfo(
                id=row[model_id_index],
                name=row[model_name_index],
                json_string=row[model_json_index],
                namespace=row[model_namespace_index],
            )

            sbvr = SBVR(
                rules=row[sbvr_rules_index].split("\n"),
                vocab=row[sbvr_vocab_index].split("\n")
            )

            descriptions = ProcessDescriptions(
                from_sbvr=LLMCompletion(
                    text=row[from_sbvr_text_index],
                    prompt_tokens=int(row[from_sbvr_prompt_tokens_index]),
                    completion_tokens=int(row[from_sbvr_completion_tokens_index]),
                ),
                from_picture=LLMCompletion(
                    text=row[from_picture_text_index],
                    prompt_tokens=int(row[from_picture_prompt_tokens_index]),
                    completion_tokens=int(row[from_picture_completion_tokens_index]),
                ),
                from_both=LLMCompletion(
                    text=row[from_both_text_index],
                    prompt_tokens=int(row[from_both_prompt_tokens_index]),
                    completion_tokens=int(row[from_both_completion_tokens_index]),
                ),
            )

            yield DescribedModel(
                model=model,
                sbvr=sbvr,
                descriptions=descriptions,
            )
