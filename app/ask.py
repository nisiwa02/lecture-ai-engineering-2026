"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 会話履歴を保って続けて質問する

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app import tools
from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _prompt_with_context(question: str) -> str:
    context = tools.retrieve_glossary(question)
    return (
        "あなたはAIE2026の学習支援アシスタントです。用語集の検索結果を根拠として回答してください。"
        "講座固有の情報が検索結果にない場合は、推測せず用語集では確認できないと伝えてください。\n\n"
        f"【用語集の検索結果】\n{context}\n\n"
        f"【質問】\n{question}"
    )


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=_prompt_with_context(question))

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）
    _print_usage(response.usage_metadata)

    return response.text


def _print_usage(usage) -> None:
    if usage is not None:
        print(
            f"[usage] input={usage.prompt_token_count} "
            f"output={usage.candidates_token_count}"
        )


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(model=MODEL)
    print("会話モードです。空欄または Ctrl-D で終了します。")
    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            break
        if not question:
            break

        response = conversation.send_message(_prompt_with_context(question))
        print(f"\n{response.text}\n")
        _print_usage(response.usage_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--interactive", "-i", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            chat()
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
