import json5
from typing import Any, Dict


def extract_res_object(model_output: str, output_key: str) -> Dict[str, Any]:
    """
    从任意字符串里尽量稳健地提取第一个包含指定 key 的 JSON 对象。
    要求：
    - 能跳过前后杂讯
    - 正确处理字符串里的大括号 / 反斜杠 / 引号
    - 只要某个 JSON 对象里包含 output_key，就返回该对象，否则返回 {}
    """
    if not isinstance(model_output, str):
        return {}

    text = model_output
    n = len(text)

    # 从每一个 '{' 位置尝试做一次“括号匹配 + json5 解析”
    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        quote_char = ""

        # 从 start 开始往后扫，做一个对大括号、字符串转义都敏感的匹配
        for i in range(start, n):
            ch = text[i]

            if in_string:
                if escape:
                    # 当前字符被转义，直接吞掉
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    # 字符串结束
                    in_string = False
            else:
                # 不在字符串里时才能改变括号层级
                if ch == '"' or ch == "'":
                    in_string = True
                    quote_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        # 提前遇到 '}'，说明这个起点不合法，放弃这个 start
                        break
                    depth -= 1
                    # 当 depth 回到 0，说明从 start 到 i 是一个完整的 JSON 对象片段
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json5.loads(candidate)
                        except Exception:
                            # 这个片段不是合法 JSON，对当前 start 无需再继续扩大范围
                            break

                        if isinstance(obj, dict) and output_key in obj:
                            return obj

                        # 这个 JSON 是合法的，但不含指定 key，
                        # 仍然可以结束当前 start（避免 O(n^2) 继续扩展）。
                        break

        # 继续尝试下一个 '{' 的位置

    return {}


if __name__ == "__main__":
    # 一些简单测试
    cases = [
        '{"answer": "asdasd\\\\boxed{123123}"}',
        '前缀噪声 {"answer": 42} 后缀噪声',
        '```json\\n{"answer": "ok"}\\n```',
        '没有目标 key 的 {"other": 1}',
        '嵌套对象 {"answer": {"inner": 1}}',
    ]

    for c in cases:
        print(c, "=>", extract_res_object(c, "answer"))
