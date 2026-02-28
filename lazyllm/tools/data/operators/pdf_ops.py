from typing import Optional
from lazyllm import pipeline, OnlineChatModule
from lazyllm.tools.data import Pdf2QA, Pdf2Qa, Text2qa


def build_pdf2qa_pipeline(
        model,
        mineru_api,

        # ===== Pdf2Md =====
        pdf_input_key: str = 'pdf_path',
        pdf_output_key: str = 'docs',
        pdf_upload_mode: bool = True,
        pdf_use_cache: bool = False,

        # ===== TextToChunks =====
        text_key: str = 'text',
        chunk_key: str = 'chunk',
        tokenizer=None,
        chunk_size: int = 100,
        tokenize: bool = False,

        # ===== PdfChunkToQA =====
        qa_input_key: str = 'chunk',
        qa_query_key: str = 'query',
        qa_answer_key: str = 'answer',
        qa_user_prompt: Optional[str] = None,
        qa_image_key: str = 'image_path',

        # ===== PdfQAScorer =====
        scorer_input_key: str = 'chunk',
        scorer_output_key: str = 'score',
        scorer_query_key: str = 'query',
        scorer_answer_key: str = 'answer',
        scorer_user_prompt: Optional[str] = None,
        scorer_image_key: str = 'image_path',

        # ===== Filter =====
        filter_input_key: str = 'score',
        threshold: float = 1,
):
    with pipeline() as ppl:

        # ========= 1️⃣ PDF → Markdown =========
        ppl.pdf2md = Pdf2Qa.Pdf2Md(
            input_key=pdf_input_key,
            output_key=pdf_output_key,
            reader_url=mineru_api,
            upload_mode=pdf_upload_mode,
            use_cache=pdf_use_cache,
        )

        # ========= 2️⃣ Text → Chunks =========
        ppl.text_to_chunks = Text2qa.TextToChunks(
            input_key=text_key,
            output_key=chunk_key,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            tokenize=tokenize
        )

        # ========= 3️⃣ Chunk → QA =========
        ppl.generate_qa = Pdf2Qa.PdfChunkToQA(
            input_key=qa_input_key,
            query_key=qa_query_key,
            answer_key=qa_answer_key,
            model=model,
            user_prompt=qa_user_prompt,
            mineru_api=mineru_api,
            image_key=qa_image_key,
        )

        # ========= 4️⃣ QA Scoring =========
        ppl.qa_scorer = Pdf2Qa.PdfQAScorer(
            input_key=scorer_input_key,
            output_key=scorer_output_key,
            query_key=scorer_query_key,
            answer_key=scorer_answer_key,
            model=model,
            user_prompt=scorer_user_prompt,
            image_key=scorer_image_key,
        )

        ppl.quality_filter = Pdf2QA.multi_features_filter(
            input_key=filter_input_key,
            threshold=threshold,
        )

    return ppl

if __name__ == '__main__':

    model = OnlineChatModule(
        source='sensenova',
        model='SenseChat-Vision'
    )

    mineru_api = 'http://10.119.30.80:20234'

    test_path = {
        'pdf_path': '/home/mnt/cuishaoting/LazyLLM/lazyllm/tools/data/pipelines/test_mineru.pdf'
    }

    ppl = build_pdf2qa_pipeline(
        model=model,
        mineru_api=mineru_api,
        text_key='content',
        chunk_size=200,
        threshold=1
    )

    result = ppl(test_path)

    print(len(result))