from lazyllm import pipeline
from lazyllm.tools.data import text_to_qa, eval_qa, filter_qa


def generate_pipline():
    with pipeline() as ppl:
        ppl.text_to_qa = text_to_qa(input_text_key='content', 
                                          output_query_key='query', 
                                          output_answer_key='answer')
        ppl.eval_qa = eval_qa(input_text_key='content', 
            input_query_key='query', 
            input_answer_key='answer', 
            output_eval_key="eval")
        ppl.filter_qa = filter_qa(input_eval_score='eval', threshold = 3)
    return ppl

def build__text2qa_pipeline(source):
    text2pa_pip = generate_pipline()
    keep, drop = text2pa_pip(source)
    print("Drop: ", drop)
    while drop:
        _, drop = text2pa_pip(drop)
        keep.extend(_)

    print(keep)


if __name__ == "__main__":
    input_chunks = [{'content' : 'sentence 1'}, 
              {'content' : 'sentence 2'}, 
              {'content' : 'sentence 3'}]
    text2pa_pip = build__text2qa_pipeline(input_chunks)
