from ..base_data import DataOperatorRegistry
import random

@DataOperatorRegistry.register(one_item=False, tag='all')
def text_to_qa(data, 
               input_text_key='content', 
               output_query_key='query', 
               output_answer_key='answer'):
    assert isinstance(data, list)
    for item in data:
        input_text = item.get(input_text_key, "")
        item[output_query_key] = f'根据{input_text}生成的query'
        item[output_answer_key] = f'根据{input_text}生成的answer'
    return data


@DataOperatorRegistry.register(one_item=False, tag='all')
def eval_qa(data, 
            input_text_key='content', 
            input_query_key='query', 
            input_answer_key='answer', 
            output_eval_key="eval"
            ):
    assert isinstance(data, list)
    for item in data:
        input_text = item.get(input_text_key, "")
        input_query = item.get(input_query_key, "")
        input_answer = item.get(input_answer_key, "")
        # print(f'根据 {input_text} {input_query} {input_answer} 打分')
        item[output_eval_key] = random.choice([1, 5])
    return data

@DataOperatorRegistry.register(one_item=False, tag='all')
def filter_qa(data, input_eval_score='eval', threshold = 3):
    assert isinstance(data, list)
    keep_list = []
    drop_list = []
    for item in data:
        qa_score = item.get(input_eval_score, "")
        if qa_score >= threshold:
            keep_list.append(item)
        else:
            drop_list.append(item)

    return keep_list, drop_list

