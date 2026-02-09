from ..base_data import data_register
import regex
from lazyllm import TrainableModule, LOG
import re
import json5
from lazyllm.thirdparty import transformers

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'
DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B'
mathQA = data_register.new_group('mathQA')


def boxed_extractor(text):
    if not isinstance(text, str):
        return None
    pattern = r'\\boxed\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    matches = regex.findall(pattern, text)
    return matches[-1].strip() if matches else None


@data_register('data.mathQA', rewrite_func='forward')
def math_answer_extractor(data, input_key='answer', output_key='math_answer'):
    assert isinstance(data, dict)
    answer = data[input_key]
    math_answer = boxed_extractor(answer)
    data[output_key] = math_answer
    return data

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


class MathAnswerGenerator(mathQA):
    def __init__(self,
                 input_key='question',
                 output_key='answer',
                 regenerate_key='regenerate',
                 model=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.regenerate_key = regenerate_key
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
        请为这个数学问题生成他的推理结果：

        问题：
        {question}

        规则：
        - 输出详细的过程
        - 最终结果使用 \\boxed{{ANSWER}}包裹

        仅输出 JSON：
        {{{{'{output_key}': 推理结果}}}}
        '''.format(
            question=data[self.input_key],
            output_key=self.output_key
        )

        response = self.model(prompt)
        res = extract_res_object(response, self.output_key)

        data[self.output_key] = res.get(self.output_key, None)
        data[self.regenerate_key] = False
        return data


class DifficultyEvaluator(mathQA):
    def __init__(self,
                 input_key='question',
                 output_key='difficulty',
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
        if data.get(self.output_key) is not None:
            return None

        prompt = '''
        请根据这个数学问题，输出这道题的难度：

        难度级别：
        - Easy : 小学级别数学题
        - Medium : 初中至高中级别数学题
        - Hard ： 大学及以上级别数学题

        问题：
        {question}

        规则：你只能输出以下三种级别：Easy、Medium、Hard

        且格式仅输出 JSON：
        {{{{'{output_key}': 难度级别}}}}
        '''.format(
            question=data[self.input_key],
            output_key=self.output_key
        )

        response = self.model(prompt)
        res = extract_res_object(response, self.output_key)

        data[self.output_key] = res.get(self.output_key, None)
        return data


@data_register(
    'data.mathQA',
    rewrite_func='forward_batch_input'
)
def DifficultyEvaluatorBatch(data, input_key='difficulty'):
    result = {}
    for entry in data:
        key = entry.get(input_key)
        if key in result:
            result[key] += 1
        else:
            result[key] = 1
    return [result]


class QualityEvaluator(mathQA):
    def __init__(self,
                 question_key='question',
                 answer_key='answer',
                 output_key='score',
                 model=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.question_key = question_key
        self.answer_key = answer_key
        self.output_key = output_key
        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

    def forward(self, data):
        if data.get(self.output_key) is not None:
            return None

        prompt = '''
        对问题和答案进行正确性检查，包括格式是否规范、语义是否合理、条件是否矛盾以及是否具备充分信息可解

        问题：
        {question}
        答案：
        {answer}
        规则：
        输出 0 为问题、答案有问题 需要重新生成
        输出 1 为没有问题

        且格式仅输出 JSON：
        {{{{'{output_key}': 0 or 1}}}}
        '''.format(
            question=data[self.question_key],
            answer=data[self.answer_key],
            output_key=self.output_key
        )

        response = self.model(prompt)
        res = extract_res_object(response, self.output_key)

        data[self.output_key] = res.get(self.output_key, None)
        return data

class DuplicateAnswerDetector(mathQA):
    def __init__(self,
                 question_key='question',
                 answer_key='answer',
                 output_key='duplicate',
                 min_repeat_len=15,
                 repeat_threshold=2,
                 periodic_min_repeat=3,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.question_key = question_key
        self.answer_key = answer_key
        self.output_key = output_key

        self.min_repeat_len = min_repeat_len
        self.repeat_threshold = repeat_threshold
        self.periodic_min_repeat = periodic_min_repeat

    def _is_periodic(self, text):
        n = len(text)
        if n < 6:
            return False
        for size in range(1, n // 2 + 1):
            if n % size != 0:
                continue

            unit = text[:size]
            if unit * (n // size) == text:
                if (n // size) >= self.periodic_min_repeat:
                    return True

        return False

    def _has_long_repeat(self, merged_text):
        seen = {}
        text_len = len(merged_text)

        for i in range(text_len - self.min_repeat_len + 1):

            substr = merged_text[i:i + self.min_repeat_len]

            if not substr.strip():
                continue

            seen[substr] = seen.get(substr, 0) + 1

            if seen[substr] >= self.repeat_threshold:
                return True

        return False

    def _sentence_repeat(self, answer):
        sentences = re.split(r'[。！？.!?\n]', answer)
        counter = {}
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            counter[s] = counter.get(s, 0) + 1
            if counter[s] >= 3:
                return True
        return False

    def forward(self, data):
        assert isinstance(data, dict)
        question = str(data.get(self.question_key, '') or '')
        answer = str(data.get(self.answer_key, '') or '')
        data[self.output_key] = False
        if not answer:
            return data

        merged = question + "\n" + answer
        if self._is_periodic(answer):
            data[self.output_key] = True
            return data

        if self._sentence_repeat(answer):
            data[self.output_key] = True
            return data

        if self._has_long_repeat(merged):
            data[self.output_key] = True
            return data

        return data

class ReasoningAnswerTokenLengthFilter(mathQA):
    def __init__(self,
                 input_key='answer',
                 max_answer_token_length=300,
                 tokenize=True,
                 tokenizer=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.max_answer_token_length = max_answer_token_length
        self.tokenizer = tokenizer

        if tokenize and tokenizer is None:
            LOG.warning(
                f'tokenize=True but tokenizer is None, '
                f'loading tokenizer from default model: {DEFAULT_TOKENIZER}'
            )
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    DEFAULT_TOKENIZER,
                    trust_remote_code=True
                )
                self.tokenize = True
            except Exception as e:
                LOG.warning(
                    f'failed to load tokenizer from {DEFAULT_TOKENIZER}, '
                    f'falling back to char count, error: {e}'
                )
                self.tokenize = False
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
            self.tokenize = tokenize

        self.empty_count = 0

    def _get_len(self, text: str):
        if text is None or (isinstance(text, str) and text.strip() == ''):
            self.empty_count += 1
            return self.max_answer_token_length + 1

        try:
            if self.tokenize:
                return len(
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False
                    )
                )
            return len(text)

        except Exception as e:
            LOG.warning(f'token encode failed: {e}')
            self.empty_count += 1
            return self.max_answer_token_length + 1

    def forward(self, data: dict):
        text = data.get(self.input_key, '')
        if not text:
            self.empty_count += 1
            return []

        token_len = self._get_len(text)

        if token_len <= self.max_answer_token_length:
            return None

        # clear eligible answer
        data[self.input_key] = ''
        return data