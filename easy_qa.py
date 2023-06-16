# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code here based on TruthfulQA code: https://huggingface.co/datasets/truthful_qa/blob/main/truthful_qa.py

import csv
import datasets
from datasets import load_dataset

_DESCRIPTION = """\
EasyQA is a GPT-3.5-turbo-generated dataset of easy kindergarten-level facts, meant to be used to prompt and evaluate large language models for "common sense" truthful responses. It was originally created to understand how different types of truthfulness may be represented in the intermediate activations of large language models. EasyQA compromises 2346 questions that span 50 categories, including art, technology, education, music, and animals. Questions are crafted to be extremely simple and obvious, eliciting an obvious truth that would not be susceptible to misconceptions.
"""

_LICENSE = "Apache License 2.0"

_HOMEPAGE = "https://huggingface.co/datasets/notrichardren/easy_qa"

_CITATION = """\
@misc{ez_QA,
    title={EasyQA: A Kindergarten-Level QA Benchmark},
    author={Kevin Wang and Richard Ren and Phillip Guo},
    year={2023},
}
"""

class TruthfulQaConfig(datasets.BuilderConfig):

    def __init__(self, url, features, **kwargs):
        super().__init__(version=datasets.Version("1.1.0"), **kwargs)
        self.url = url
        self.features = features

class TruthfulQa(datasets.GeneratorBasedBuilder):
    """TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions."""

    BUILDER_CONFIGS = [
        TruthfulQaConfig(
            name="easy_qa",
            url="easy_facts.csv",
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "right_answer": datasets.Value("string"),
                    "wrong_answer": datasets.Value("string"),
                    "category": datasets.Value("string"),
                }
            ),
            description="easy_qa",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download(self.config.url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _split_csv_list(self, csv_list: str, delimiter: str = ";") -> str:
        """
        Splits a csv list field, delimited by `delimiter` (';'), into a list
        of strings.
        """
        csv_list = csv_list.strip().split(delimiter)
        return [item.strip() for item in csv_list]

    def _generate_examples(self, filepath):
        # Generation data is in a `CSV` file.
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            contents = csv.DictReader(f)
            for key, row in enumerate(contents):
                # Ensure that references exist.
                row
                if not row["Right"] or not row["Wrong"]:
                    continue
                yield key, {
                    "question": row["Question"],
                    "right_answer": row["Right"],
                    "wrong_answer": row["Wrong"],
                    "category": row["Category"],
                }