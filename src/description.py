import abc
import base64
import pathlib
import typing

import dotenv
import openai
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionContentPartTextParam, \
    ChatCompletionContentPartImageParam, ChatCompletionContentPartParam, ChatCompletionDeveloperMessageParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

import load
import prompts

dotenv.load_dotenv()


class BaseLLMDescriber(abc.ABC):
    def __init__(self, client: openai.OpenAI, model: str):
        self.client = client
        self.model = model
        self._prompt_dir = pathlib.Path(__file__).parent.parent / "resources" / "prompts"
        self._prompt_template = self.get_prompt_template()

    @abc.abstractmethod
    def get_prompt_template(self) -> prompts.Prompt:
        raise NotImplementedError()

    def request_params(
            self,
            *,
            image_path: typing.Optional[pathlib.Path | str],
            sbvr: typing.Optional[str],
            example: typing.Optional[str],
            reasoning_effort: typing.Literal["minimal", "low", "medium", "high"] = "minimal"
    ):
        prompt = self._prompt_template.apply()

        content: typing.List[ChatCompletionContentPartParam | str] = [
            ChatCompletionContentPartTextParam(
                text=prompt,
                type="text"
            )
        ]
        if example is not None:
            content.append(ChatCompletionContentPartTextParam(
                text=example,
                type="text"
            ))

        if image_path is not None:
            if sbvr is not None:
                content.append(ChatCompletionContentPartTextParam(
                    text="You are also given this image of the process to help you, "
                         "please do not describe the process twice, but integrate the "
                         "information in the image and SBVR rules into one description.",
                    type="text"
                ))
            else:
                content.append(ChatCompletionContentPartTextParam(
                    text="The image of the process is as follows:",
                    type="text")
                )
            with open(image_path, "rb") as image_file:
                b64_image = base64.b64encode(image_file.read()).decode("utf-8")
            content.append(
                ChatCompletionContentPartImageParam(
                    image_url=ImageURL(url=f"data:image/png;base64,{b64_image}"),
                    type="image_url"
                )
            )
        if sbvr is not None:
            content.append(ChatCompletionContentPartTextParam(
                text=f"These are the SBVR rules in question:\n\n{sbvr}",
                type="text"
            ))

        return {
            "messages": [
                ChatCompletionDeveloperMessageParam(
                    role="instruction",
                    content=[]
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=content
                )
            ],
            "model": self.model,
            "reasoning_effort": reasoning_effort
        }

    def describe(
            self,
            *,
            image_path: typing.Optional[pathlib.Path | str],
            sbvr: typing.Optional[str],
            example: typing.Optional[str],
            reasoning_effort: typing.Literal["minimal", "low", "medium", "high"] = "minimal"
    ) -> load.LLMCompletion:
        request_params = self.request_params(
            image_path=image_path,
            sbvr=sbvr,
            example=example,
            reasoning_effort=reasoning_effort
        )
        resp = self.client.chat.completions.create(
            **request_params
        )

        return load.LLMCompletion(text=resp.choices[0].message.content,
                                  prompt_tokens=resp.usage.prompt_tokens,
                                  completion_tokens=resp.usage.completion_tokens)


class LLMPictureDescriber(BaseLLMDescriber):
    def get_prompt_template(self) -> prompts.Prompt:
        return prompts.Prompt(self._prompt_dir / "describe-image.txt")


class LLMCombinedDescriber(BaseLLMDescriber):
    def get_prompt_template(self) -> prompts.Prompt:
        return prompts.Prompt(self._prompt_dir / "describe-combined.txt")


class LLMSBVRDescriber(BaseLLMDescriber):
    def get_prompt_template(self) -> prompts.Prompt:
        return prompts.Prompt(self._prompt_dir / "describe-sbvr.txt")


if __name__ == "__main__":
    def main():
        model = "gpt-5-mini-2025-08-07"
        # model = "gpt-5-nano-2025-08-07"
        describer = LLMPictureDescriber(client=openai.OpenAI(), model=model)
        plausible_models_path = pathlib.Path(__file__).parent.parent / "resources" / "plausible" / "models"
        img_path = plausible_models_path / "0" / "images" / "Week 4 Task 2.png"
        describer.describe(img_path)


    main()
