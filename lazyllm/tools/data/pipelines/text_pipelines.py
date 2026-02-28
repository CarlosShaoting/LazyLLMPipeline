from lazyllm import pipeline, OnlineChatModule
from lazyllm.tools.data import Text2qa


def build_text2qa_pipeline(
        text_key='text',
        chunk_key='chunk',
        instruction_key='instruction',
        output_key='output',
        model=None,
        user_prompt=None,
        tokenizer=None,
        chunk_size=100,
        tokenize=False,
        threshold=1):

    if model is None:
        model = OnlineChatModule()

    with pipeline() as ppl:

        ppl.text_to_chunks = Text2qa.TextToChunks(
            input_key=text_key,
            output_key=chunk_key,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            tokenize=tokenize
        )

        ppl.noise_filter = Text2qa.empty_or_noise_filter(
            input_key=chunk_key
        )

        ppl.invalid_unicode_cleaner = Text2qa.invalid_unicode_cleaner(
            input_key=chunk_key
        )

        ppl.generate_qa = Text2qa.ChunkToQA(
            input_key=chunk_key,
            query_key=instruction_key,
            answer_key=output_key,
            model=model
        )

        ppl.qa_scorer = Text2qa.QAScorer(
            input_key=chunk_key,
            output_key='score',
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            user_prompt=user_prompt
        )

        ppl.score_filter = Text2qa.qa_score_filter(
            input_key='score',
            min_score=threshold
        )

        ppl.sft_data = Text2qa.to_alpaca_sft(
            query_key=instruction_key,
            context_key=chunk_key,
            answer_key=output_key
        )

    return ppl


if __name__ == '__main__':

    story = (
        '夜里十一点，旧书店还亮着灯。林默推门进来时，风铃轻轻一响，'
        '像是提醒某段被遗忘的时光。店里只剩老板娘，她总说书会挑人，而不是人挑书。\n\n'
        '林默最近总做同一个梦：在一座看不见尽头的桥上奔跑，桥下是翻涌的雾。'
        '他不知道自己在追什么，只知道不能停。老板娘递给他一本没有书名的旧书，'
        '说：“也许它在等你。”封面泛黄，边角卷起，像被许多人握过。\n\n'
        '他翻开第一页，里面写着一句话——“当你不再逃跑，桥就会出现终点。”'
        '林默愣住了。那一瞬间，他忽然明白，自己害怕的不是未来，而是承认平凡的勇气。\n\n'
        '窗外的风停了。林默合上书，深吸一口气。桥依旧在梦里延伸，但这一次，他决定慢慢走。'
    )

    story2 = (
        '清晨六点，海边的小镇还笼在薄雾里。阿远骑着旧自行车穿过石板路，'
        '车铃声在空荡的街道上回响。他每天都会去灯塔下坐一会儿，看海浪一层层推向岸边，'
        '像时间轻轻拍打记忆。\n\n'
        '他曾想离开这里，去更远的城市闯荡，可真正站在车站时，却总想起母亲在院子里晾衣服的背影，'
        '想起晚饭时窗外橘色的夕阳。原来人不是被困在原地，而是被温柔牵住。\n\n'
        '那天，他把一封写好的辞职信撕碎，任碎纸随风飞散。海风带着咸味扑在脸上，他忽然觉得安心。'
        '也许远方并不只在地图上，有时候，学会珍惜脚下的土地，本身就是一次勇敢的远行。'
    )

    input_chunks = [
        {'text': story},
        {'text': story2}
    ]

    model = OnlineChatModule()
    text2qa_pip = build_text2qa_pipeline(model=model)

    res = text2qa_pip(input_chunks)

    print(len(res))
