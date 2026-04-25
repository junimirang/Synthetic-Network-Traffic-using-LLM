# -*- coding: utf-8 -*-
"""
input_helper.py
공통 대화형 입력 헬퍼 — 모든 프로그램에서 import
"""

def ask(prompt, choices=None, default=None, cast=str):
    """
    대화형 입력 함수
      choices : 허용 값 목록 (None이면 제한 없음)
      default : 엔터 입력 시 기본값
      cast    : 타입 변환 (str / int / float)
    """
    choice_str = f" [{'/'.join(choices)}]" if choices else ""
    default_str = f" (기본값: {default})" if default is not None else ""
    full_prompt = f"  {prompt}{choice_str}{default_str}: "

    while True:
        raw = input(full_prompt).strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            val = cast(raw)
        except ValueError:
            print(f"  ※ 올바른 값을 입력하세요.")
            continue
        if choices and str(val) not in choices:
            print(f"  ※ 선택 가능한 값: {choices}")
            continue
        return val


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")
