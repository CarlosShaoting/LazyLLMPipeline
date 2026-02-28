import os
import re
import json
from lazyllm import OnlineChatModule
from lazyllm.components.formatter import encode_query_with_filepaths
# 初始化 VLM 模型（SenseNova）
model = OnlineChatModule(source='sensenova', model='SenseChat-Vision')

# 测试 chunk
chunk = "![Man what can I say. ](images/8aa0ff33eebaefeb51735691c1a35ec56ac6cfbd6487c480c3dffce1497084d2.jpg)"
mineru_api = "http://10.119.30.80:20234"  # api base

# 提取图片
pattern = r'!\[.*?\]\((images/[^)]+)\)'
image_rel_paths = re.findall(pattern, chunk)
full_paths = [os.path.join(mineru_api, p) for p in image_rel_paths]

# 文本上下文
context = re.sub(pattern, '', chunk).strip()
title = "测试文件"

print(model(encode_query_with_filepaths('这是什么', ['http://10.119.30.80:20234/images/8aa0ff33eebaefeb51735691c1a35ec56ac6cfbd6487c480c3dffce1497084d2.jpg'])))