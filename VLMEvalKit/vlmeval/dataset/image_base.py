import pandas as pd
from abc import abstractmethod
from ..smp import *
import re

def img_root_map(dataset):
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if 'MMMU' in dataset:
        return 'MMMU'
    if "QSpatial" in dataset:
        return "QSpatial"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset


class ImageBaseDataset:

    MODALITY = 'IMAGE'
    DATASET_URL = {}
    DATASET_MD5 = {}

    def __init__(self, dataset='MMBench', skip_noimg=True):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', img_root_map(dataset))

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL[dataset]
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # 定义替换规则列表






    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        # instruction = "Instruction: Put the final value in \\boxed{}.\n"
        # question = re.sub(r"(Hint:.*?)\n", r"\1\n" + instruction, line['question'], count=1)
        # question =  re.sub(r"(at the end)\.", r"in boxed{} \1.", line['question'], count=1)
        # question = instruction+line['question']
        # question = line['query_cot']
        def update_hint_format(text):
            # 定义要匹配和替换的模式与格式
            replacements = [
                (
                    r"Hint: Please answer the question and provide the correct option letter, e\.g\.,\s*([A-D]),\s*([A-D]),\s*([A-D]),\s*([A-D]),\s*at the end\.",
                    r"Hint: Please answer the question step by step and provide the correct option letter in \\boxed{()}, e.g., \\boxed{(\1)}, \\boxed{(\2)}, \\boxed{(\3)}, \\boxed{(\4)}, at the end."
                ),

                (
                    r"Hint: Please answer the question requiring an integer answer and provide the final value, e\.g\.,\s*(\d+),\s*(\d+),\s*(\d+),\s*at the end\.",
                    r"Hint: Please answer the question requiring an integer answer step by step and provide the final value in \\boxed{}, e.g., \\boxed{\1}, \\boxed{\2}, \\boxed{\3}, at the end."
                ),

                (
                    r"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e\.g\.,\s*([\d]+\.[\d]),\s*([\d]+\.[\d]),\s*([\d]+\.[\d]),\s*at the end\.",
                    r"Hint: Please answer the question requiring a floating-point number with one decimal place step by step and provide the final value in \\boxed{}, e.g., \\boxed{\1}, \\boxed{\2}, \\boxed{\3}, at the end."
                ),

                (
                    r"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e\.g\.,\s*([\d]+\.[\d]{2}),\s*([\d]+\.[\d]{2}),\s*([\d]+\.[\d]{2}),\s*at the end\.",
                    r"Hint: Please answer the question requiring a floating-point number with two decimal places step by step and provide the final value in \\boxed{}, e.g., \\boxed{\1}, \\boxed{\2}, \\boxed{\3}, at the end."
                ),

                (
                    r"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e\.g\.,\s*(\[[^\]]+\]),\s*(\[[^\]]+\]),\s*at the end\.",
                    r"Hint: Please answer the question requiring a Python list as an answer step by step and provide the final list in \\boxed{}, e.g., \\boxed{\1}, \\boxed{\2}, at the end."
                ),
            ]            

            for pattern, repl in replacements:
                text = re.sub(pattern, repl, text)

            return text
        def change_prompt_MathVerse_MINI(text):
            replacements = [
                # 替换 e.g., A, B, C, D → \boxed{A}, ...
                (
                    r"e\.g\.,\s*([A-D]),\s*([A-D]),\s*([A-D]),\s*([A-D])",
                    r"e.g., \\boxed{\1}, \\boxed{\2}, \\boxed{\3}, \\boxed{\4}"
                ),
                # 替换 e.g., 1, 2.5, 300. → \boxed{1}, \boxed{2.5}, \boxed{300}.
                (
                    r"e\.g\.,\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\.",
                    r"e.g., \\boxed{\1}, \\boxed{\2}, \\boxed{\3}."
                ),
                (
                    r"directly answer the question",
                    r"answer the question stpe by step"
                )
            ]

            for pattern, repl in replacements:
                text = re.sub(pattern, repl, text)

            return text
        if self.dataset_name=='MathVista_MINI':
            question=update_hint_format(line['question'])
        elif "MathVerse_MINI" in self.dataset_name:
            question=change_prompt_MathVerse_MINI(line['question'])
            

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass
