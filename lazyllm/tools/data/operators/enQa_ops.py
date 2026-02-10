from ..base_data import data_register
import regex
from lazyllm import TrainableModule, LOG
import re
import json5
from lazyllm.thirdparty import transformers

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'
DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B'

enQA = data_register.new_group('enQA')

def extract_res_object(model_output,output_key):
    if isinstance(output_key, str):
        required_keys = [output_key]
    else:
        required_keys = [str(k) for k in output_key]
        if not required_keys:
            return {}

    text = model_output
    n = len(text)

    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        quote_char = ""

        for i in range(start, n):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
            else:
                if ch == '"' or ch == "'":
                    in_string = True
                    quote_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        break
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json5.loads(candidate)
                        except Exception:
                            break

                        if isinstance(obj, dict) and all(
                            k in obj for k in required_keys
                        ):
                            return obj
                        break
    return {}

class QueryRewriter(enQA):

    def __init__(self,
                 input_key='query',
                 output_key='rewrite_querys',
                 rewrite_num=3,
                 model=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.rewrite_num = rewrite_num

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

    def forward(self, data):

        query = data.get(self.input_key)
        if not query:
            return None

        if data.get(self.output_key) is not None:
            return None

        prompt = '''
        请重写下面的问题，使其语义一致但表达不同。

        原问题：
        {query}

        规则：
        - 生成 {num} 个不同表达
        - 保持语义一致
        - 不要解释

        仅输出 JSON：
        {{{{'{output_key}': [重写后的问题列表]}}}}
        '''.format(
            query=query,
            num=self.rewrite_num,
            output_key=self.output_key
        )

        response = self.model(prompt)

        res = extract_res_object(response, self.output_key)

        data[self.output_key] = res.get(self.output_key, [])
        return data

class DiversityScorer(enQA):

    def __init__(self,
                 input_key='rewrite_querys',
                 output_key='diversity_querys',
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
        querys = data.get(self.input_key)
        if not querys:
            return None

        if data.get(self.output_key) is not None:
            return None

        prompt = '''
        判断下面问题列表的表达多样性。

        问题列表：
        {querys}

        规则：
        - 表达重复或相似度高：score = 0
        - 表达差异明显：score = 1
        - 输出与输入顺序一致

        仅输出 JSON：
        {{{{'diversity_scores': [0或1列表]}}}}
        '''.format(querys=querys)

        response = self.model(prompt)

        res = extract_res_object(response, 'diversity_scores')

        scores = res.get('diversity_scores', [])

        new_list = []
        for i, q in enumerate(querys):
            score = scores[i] if i < len(scores) else 0
            new_list.append({
                'rewritten_query': q,
                'diversity_score': score
            })

        data[self.output_key] = new_list
        return data


class PostProcessor(enQA):

    def __init__(self,
                 input_key='diversity_querys',
                 rewritten_key='query',
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.rewritten_key = rewritten_key

    def forward(self, data):
        items = data.get(self.input_key)
        if not items:
            return None

        result = []
        for obj in items:

            if not isinstance(obj, dict):
                continue

            new_row = data.copy()
            new_row.pop(self.input_key, None)
            # extract the rewritten query and diversity score
            for k, v in obj.items():
                new_row[k] = v

            result.append(new_row)

        return result

class DiversityFilter(enQA):

    def __init__(self,
                 input_key='diversity_score',
                 min_score=1,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.min_score = min_score

    def forward(self, data):
        score = data.get(self.input_key)
        if score >= self.min_score:
            return None
        return []
