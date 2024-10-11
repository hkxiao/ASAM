import pandas as pd
from huggingface_hub import hf_hub_url
import datasets
import os

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "label_dir": datasets.Value("string"),
        "text": datasets.Value("string"),
        "filename": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class Fill50k(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": '../sam-1b/sa_000000-controlnet-train.jsonl',
                    "images_dir": '../sam-1b/sa_000000',
                    "conditioning_images_dir": '../sam-1b/sa_000000',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": '../sam-1b/sa_000138-controlnet-validation.json',
                    "images_dir": '../sam-1b/sa_000138',
                    "conditioning_images_dir": '../sam-1b/sa_000138',
                },
            ),
        ]
    
    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["prompt"]

            image_path = row["target"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["source"]
            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["source"]
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["target"], {
                "text": text,
                "label_dir": image_path[:-4],
                'filename': image_path[:-4].split('/')[-1],
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }