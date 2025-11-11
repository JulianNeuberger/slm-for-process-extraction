import csv
import io
import pathlib

import tqdm
from PIL import Image

import load
from sapsam.RepresentationGenerator import RepresentationGenerator

resources_dir = pathlib.Path(__file__).parent.parent / "resources"


def create_model(model_name: str, model_json: str, model_namespace: str, store: bool = True):
    gen = RepresentationGenerator()
    image_bytes = gen.generate_image(model_name, model_json, model_namespace, deletes=not store)
    return Image.open(io.BytesIO(image_bytes))


def save_model_image(
        image_directory: pathlib.Path,
        *,
        model_info: load.ModelInfo,
        overwrite=False,
        store=True
):
    image_directory.mkdir(parents=True, exist_ok=True)
    image_file_path = image_directory / f"{model_info.id}.png"
    if image_file_path.exists() and not overwrite:
        return

    image = create_model(model_info.name, model_info.json_string, model_info.namespace, store=store)

    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    background.save(image_file_path)


def draw_models_from_file(*,
                          in_file: pathlib.Path,
                          image_directory: pathlib.Path,
                          overwrite: bool = False,
                          store: bool = False):
    for model in tqdm.tqdm(load.load_raw_models(in_file)):
        save_model_image(image_directory=image_directory,
                         model_info=model,
                         overwrite=overwrite, store=store)
