from lazyllm import pipeline, OnlineChatModule
from lazyllm.tools.data import enQA, Pdf2QA, PT_MM, Pdf2Qa, Text2qa

def build_pdf2qa_pipeline(
        model,
        mineru_api,
        text_key='text',
        chunk_key='chunk',
        instruction_key='instruction',
        output_key='output',
        user_prompt=None,
        tokenizer=None,
        chunk_size=100,
        tokenize=False,
        threshold=1
        ):
    with pipeline() as ppl:
        ppl.pdf2md = Pdf2Qa.Pdf2Md(reader_url=mineru_api)

        ppl.text_to_chunks = Text2qa.TextToChunks(
            input_key=text_key,
            output_key=chunk_key,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            tokenize=tokenize
        )
        

        ppl.generate_qa = Pdf2Qa.PdfChunkToQA(
            input_key=chunk_key,
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            mineru_api=mineru_api
        )

        ppl.qa_scorer = Pdf2Qa.PdfQAScorer(
            model=model,
            input_key=chunk_key,
            query_key=instruction_key,
            answer_key=output_key
        )
    return ppl


model = OnlineChatModule(source='sensenova', model='SenseChat-Vision')
mineru_api = 'http://10.119.30.80:20234'
# text2qa_ppl = build_text2qa_pipeline(text_key='content', model=model)
test_path = {'pdf_path': '/home/mnt/cuishaoting/LazyLLM/lazyllm/tools/data/pipelines/test_mineru.pdf'}
ppl = build_pdf2qa_pipeline(model=model, mineru_api=mineru_api, text_key='content')

print(len(ppl(test_path)))

