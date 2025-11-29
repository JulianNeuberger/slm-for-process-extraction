import pathlib
import json
import typing

if __name__ == "__main__":
    def main():
        resources_dir = pathlib.Path(__file__).parent.parent / "resources"
        batches_dir = resources_dir / "batches-small" / "descriptions" / "outputs"
        num_prompt_tokens: typing.Dict[str, typing.List[int]] = {}
        for file_path in batches_dir.iterdir():
            with open(file_path, "r") as f:
                for line in f:
                    batch = json.loads(line)
                    mode = batch["custom_id"].split("-")[-1]
                    if mode not in num_prompt_tokens:
                        num_prompt_tokens[mode] = []
                    num_prompt_tokens[mode].append(batch["response"]["body"]["usage"]["prompt_tokens"])
        num_docs = len(list(num_prompt_tokens.values())[0])
        assert all(len(n) == num_docs for n in num_prompt_tokens.values())
        print(f"Described {num_docs} documents.")
        for mode, num in num_prompt_tokens.items():
            print(f"{mode}: {sum(num) / len(num)}")

    main()