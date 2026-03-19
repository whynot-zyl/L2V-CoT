from __future__ import annotations
import re

class Qwen2VLPromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        # MCQ_CN_PROMPT = '请直接回答选项字母。'
        # MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'
        MCQ_CN_PROMPT = '逐步思考问题，并将最终答案放到 \\boxed{}.\n'
        MCQ_EN_PROMPT ="Answer the following question step by step and put the final answer in \\boxed{}.\n"

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset=='MathVista_MINI':
            question = self.change_prompt(question)
        if 'MathVerse_MINI' in dataset:
            question = self.change_prompt_MathVerse_MINI(question)
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        # YORN_PROMPT = ' Please answer yes or no.'
        YORN_PROMPT = "Answer the following question step by step and put the final answer in \\boxed{}.\n"

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset=='MathVista_MINI':
            question = self.change_prompt(question)
        if 'MathVerse_MINI' in dataset:
            question = self.change_prompt_MathVerse_MINI(question)
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        # VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'        
        VQA_PROMPT = "Answer the following question step by step and put the final answer in \\boxed{}.\n"

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset=='MathVista_MINI':
            question = self.change_prompt(question)
        if 'MathVerse_MINI' in dataset:
            question = self.change_prompt_MathVerse_MINI(question)
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs
    def change_prompt(self, text):
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
    def change_prompt_MathVerse_MINI(self, text):
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
            # (
            #     r"directly answer the question",
            #     r"answer the question stpe by step"
            # )
        ]

        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)

        return text