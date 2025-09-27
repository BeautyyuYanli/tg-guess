from typing import Literal
from prompt_bottle import render
from pydantic_ai import Agent, ModelRetry, TextOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from qwq_tag import QwqTag
from pathlib import Path

PROMPT = """
你正在和用户玩猜谜游戏。
你会获得一个文档，每回合由你给出一个来自该文档的提示，用户会预测一个答案。
提示是一句简短的句子，不得正面描述文档的主题，而是从细节处给出提示，提示也应当尽量隐晦，不可以是人尽皆知的常识。
你要判断这个答案是否正确（如果用户的回答是该文档的主题，比如文档的标题或别名，即为正确）
回复格式为以下两种 xml tag 之一:

<CORRECT>文档的总结</CORRECT>
<NEXT comment="戏谑地评价用户的答案，注意不要给出实质性提示">给出下一句提示</NEXT>

第一轮的回复总为 <NEXT> tag，不含 comment 属性。
如果用户表示投降放弃，则使用 <CORRECT> tag。
如果用户试图偏离猜谜游戏的游戏进程，则使用 <NEXT> tag，并给出狠狠嘲讽的 comment。

<Document>
{{ document }}
</Document>
<History>
{% for hint, guess in rounds%}
hint: <NEXT>{{ hint }}</NEXT>
guess: {{ guess }}
{% endfor %}
</History>
"""

def parse_answer(text: str) -> tuple[Literal["CORRECT", "NEXT"], str]:
    res = QwqTag.from_str(text)
    try:
        only_res = [
            x for x in res if isinstance(x, QwqTag) and x.name in ["CORRECT", "NEXT"]
        ][0]
    except Exception as e:
        raise ModelRetry("Output Format Error") from e

    return only_res.name, only_res.attr.get("comment", "") + "\n" + only_res.content_text # type: ignore

    

async def guess(llm: OpenAIChatModel, document: str, rounds: list[tuple[str, str]]) -> tuple[Literal["CORRECT", "NEXT"], str]:
    prompt = render(PROMPT, document=document, rounds=rounds)
    agent = Agent(
        model=llm,
        output_type=TextOutput(parse_answer),
        output_retries=3,
    )
    return (await agent.run(message_history=prompt)).output





def init_llm() -> OpenAIChatModel:
    from dotenv import load_dotenv
    import logfire

    load_dotenv()
    logfire.configure()
    logfire.instrument_pydantic_ai()

    return OpenAIChatModel(
        model_name="qwen/qwen3-235b-a22b-2507",
        provider=OpenRouterProvider(),
    )

async def main() -> None:
    from pathlib import Path
    from vibe_emp.gameplay import get_output_dir, read_candidates, choose_file, play

    llm: OpenAIChatModel = init_llm()
    output_dir: Path = get_output_dir()

    candidates: list[tuple[str, int, Path]] = read_candidates(output_dir)
    _, _, chosen_file = choose_file(candidates)

    with chosen_file.open("r", encoding="utf-8") as f:
        document: str = f.read()

    await play(document=document, llm=llm, fail_label=chosen_file.name)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


    

