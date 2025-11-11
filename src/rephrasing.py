import pathlib

import openai
import dotenv
from openai.types.chat import ChatCompletionUserMessageParam

import load
import prompts

dotenv.load_dotenv()


class LLMRephraser:
    def __init__(self, client: openai.OpenAI, model: str):
        self.client = client
        self.model = model
        prompt_dir = pathlib.Path(__file__).parent.parent / "resources" / "prompts"
        self.prompt_template = prompts.Prompt(prompt_dir / "describe-sbvr.txt")

    def rephrase(self, *, text: str, example: str) -> load.LLMCompletion:
        prompt = self.prompt_template.apply(text=text, example=example)
        resp = self.client.chat.completions.create(
            messages=[
                ChatCompletionUserMessageParam(
                    role="user",
                    content=prompt
                )
            ],
            model=self.model,
        )

        return load.LLMCompletion(
            text=resp.choices[0].message.content,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )


if __name__ == "__main__":
    def main():
        sbvr = (
            """It is obligatory that Accounting department Send invoice after the process starts
    It is obligatory that Business unit Receive invoice after Accounting department Send invoice
    It is obligatory that Business unit Approve invoice after Business unit Receive invoice
    It is obligatory that either Business unit Send approved invoice in case Yes or Business unit Inform accounting department in case No but only one after Business unit Approve invoice
    It is obligatory that Accounting department Inform supplier after Business unit Inform accounting department
    It is obligatory that Accounting department Receive approved invoice after Business unit Send approved invoice
    It is obligatory that Accounting department Fill in accounting information after Accounting department Receive approved invoice
    It is obligatory that the process ends after Accounting department Inform supplier
    It is obligatory that Accounting department Pay supplier after Accounting department Fill in accounting information
    It is obligatory that Accounting department Update accounts payable after Accounting department Fill in accounting information
    It is obligatory that the process ends after all """
        )
        prompt_dir = pathlib.Path(__file__).parent.parent / "resources" / "prompts"
        with open(prompt_dir / "examples" / "pet-example.txt") as f:
            example = f.read()
        model = "gpt-5-mini-2025-08-07"
        # model = "gpt-5-nano-2025-08-07"
        rephraser = LLMRephraser(client=openai.OpenAI(), model=model)
        rephraser.rephrase(text=sbvr, example=example)
    main()
