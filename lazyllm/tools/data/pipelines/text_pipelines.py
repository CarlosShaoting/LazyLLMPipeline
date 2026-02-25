from lazyllm import pipeline, OnlineChatModule
from lazyllm.tools.data import Text2qa

def build_text2qa_pipline(text_key, chunk_key, instruction_key, output_key):
    llm = OnlineChatModule()
    with pipeline() as ppl:
        ppl.text_to_chunks = Text2qa.TextToChunks(input_key=text_key,
                                                  output_key=chunk_key,
                                                  tokenizer=None,
                                                  chunk_size=100,
                                                  tokenize=True,
                                                  )
        ppl.noise_filter = Text2qa.empty_or_noise_filter(input_key=chunk_key)
        ppl.invalid_unicode_cleaner = Text2qa.invalid_unicode_cleaner(input_key=chunk_key)
        ppl.generate_qa = Text2qa.ChunkToQA(input_key=chunk_key, query_key=instruction_key, answer_key=output_key, model=llm)

    pass

if __name__ == "__main__":
    input_chunks = [{'content' : 'sentence 1'}, 
              {'content' : 'sentence 2'}, 
              {'content' : 'sentence 3'}]
    # text2pa_pip = build__text2qa_pipeline(input_chunks)