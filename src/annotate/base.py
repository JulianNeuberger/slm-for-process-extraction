import abc
import base64
import dataclasses
import pathlib
import typing

import dotenv
import openai
from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, \
    ChatCompletionUserMessageParam, ChatCompletionDeveloperMessageParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

import data
import prompts

dotenv.load_dotenv()


@dataclasses.dataclass
class LLMAnnotation:
    doc: data.PetDocument
    prompt_tokens: int
    completion_tokens: int


class BaseParser(abc.ABC):
    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        raise NotImplementedError()


class BaseAnnotator(abc.ABC):
    def __init__(self, client: openai.OpenAI, model: str,
                 reasoning_effort: typing.Literal["minimal", "low", "medium", "high"] = "minimal"):
        self.client = client
        self.model = model
        self.prompt_template = self.get_prompt_template()
        self.text_formatter = self.get_text_formatter()
        self.parser = self.get_parser()
        self.reasoning_effort = reasoning_effort

    def get_prompt_template(self) -> prompts.Prompt:
        raise NotImplementedError()

    def get_text_formatter(self) -> typing.Callable[[data.PetDocument], str]:
        raise NotImplementedError()

    def get_parser(self) -> BaseParser:
        raise NotImplementedError()

    def get_params(
            self,
            *,
            doc: data.PetDocument,
            hints: typing.Optional[str] = None,
            image_path: typing.Optional[pathlib.Path | str] = None
    ):
        text = self.text_formatter(doc)

        prompt = self.prompt_template.apply()

        dev_content: typing.List = [
            ChatCompletionContentPartTextParam(
                text=prompt,
                type="text"
            )
        ]

        user_content = []
        if hints is not None:
            user_content.append(ChatCompletionContentPartTextParam(
                text=f"Use the following SBVR as guidance in your task. You can find the original "
                     f"activities (<activity>), actors (<actor>), and conditions (<cond>) marked with "
                     f"XML-style tags to help you identifying relevant mentions.:\n{hints}",
                type="text"
            ))
        if image_path is not None:
            user_content.append(ChatCompletionContentPartTextParam(
                text="You are also given this image the description is based on to help you.",
                type="text"
            ))
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append(ChatCompletionContentPartImageParam(
                image_url=ImageURL(url=f"data:image/png;base64,{image_b64}"),
                type="image_url"
            ))
        user_content.append(ChatCompletionContentPartTextParam(
            text=f"Here is the text in question: \n\n{text}",
            type="text"
        ))

        return {
            "messages": [
                ChatCompletionDeveloperMessageParam(
                    role="developer",
                    content=dev_content
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=user_content
                )
            ],
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }

    def batch_line(
            self,
            *,
            doc: data.PetDocument,
            hints: typing.Optional[str] = None,
            image_path: typing.Optional[pathlib.Path | str] = None
    ) -> typing.Dict:
        task_id = f"{self.__class__.__name__}-{doc.id}"
        if hints is not None and image_path is not None:
            task_id += "-combined"
        elif hints is not None:
            task_id += "-sbvr"
        elif image_path is not None:
            task_id += "-image"
        else:
            task_id += "-no_hints"

        params = self.get_params(doc=doc, hints=hints, image_path=image_path)
        return {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                **params
            }
        }

    def annotate(
            self,
            *,
            doc: data.PetDocument,
            hints: typing.Optional[str] = None,
            image_path: typing.Optional[pathlib.Path | str] = None
    ) -> LLMAnnotation:
        resp = self.client.chat.completions.create(
            **self.get_params(doc=doc, hints=hints, image_path=image_path)
        )
        answer = resp.choices[0].message.content
        doc = self.parser.parse(doc, answer)
        return LLMAnnotation(
            doc=doc,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
