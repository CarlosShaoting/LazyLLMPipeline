from ..base_data import data_register
from res_extractor import json_res_extractor, boxed_res_extractor
from lazyllm import TrainableModule, LOG
from collections import Counter, defaultdict

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'
DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B'
genCot = data_register.new_group('genCot')


class CoTGenerator(genCot):
    def __init__(self,
                 input_key='query',
                 output_key='cot_answer',
                 model=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

    def forward(self, data):
        answer = data.get(self.output_key)
        regenerate = data.get(self.regenerate_key, False)

        if answer is not None and regenerate is False:
            return None

        prompt = '''
        请为这个问题生成带有思维链（Chain-of-Thought, CoT）的输出结果：

        问题：
        {question}

        规则：
        - 输出详细的CoT过程
        - 最终结果使用 \\boxed{{ANSWER}}包裹

        仅输出 JSON：
        {{{{'{output_key}': 推理结果}}}}
        '''.format(
            question=data[self.input_key],
            output_key=self.output_key
        )

        response = self.model(prompt)
        res = json_res_extractor(response, self.output_key)

        data[self.output_key] = res.get(self.output_key, None)
        return data

class SelfConsistencyCoTGenerator(genCot):
    def __init__(self,
                 input_key='query',
                 output_key='cot_answer',
                 num_samples=5,
                 model=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.num_samples = num_samples

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

    def forward(self, data):
        question = data[self.input_key]

        prompt = '''
        请为这个问题生成带有思维链（Chain-of-Thought, CoT）的输出结果：

        问题：
        {question}

        规则：
        - 输出详细的CoT过程
        - 最终结果使用 \\boxed{{ANSWER}}包裹
        '''.format(question=question)

        cot_list = []
        boxed_answers = []

        for _ in range(self.num_samples):
            response = self.model(prompt)

            cot = response
            boxed = boxed_res_extractor(response)

            if boxed is not None:
                cot_list.append(cot)
                boxed_answers.append(boxed)

        if not boxed_answers:
            data[self.output_key] = None
            return data

        counter = Counter(boxed_answers)
        majority_answer = counter.most_common(1)[0][0]

        for cot, ans in zip(cot_list, boxed_answers):
            if ans == majority_answer:
                data[self.output_key] = cot
                return data

        data[self.output_key] = None
        return data



