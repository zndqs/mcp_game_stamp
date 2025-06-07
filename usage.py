import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langgraph_codeact import create_codeact
from langgraph.checkpoint.memory import MemorySaver

from src.game_stamp.server import generate_image

import builtins
import contextlib
import io
from typing import Any

load_dotenv()


def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        print(f'runing code-----------\n{code}')
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
        print(f'running result----------\n{result}')
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars


tools = [generate_image]
model = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="anthropic/claude-sonnet-4",
)

code_act = create_codeact(model, tools, eval)
agent = code_act.compile(checkpointer=MemorySaver())

messages = [{
    "role": "user",
    "content": "生成一个html游戏，屏幕上有很多堆叠非常多层的方块，玩家需要点击屏幕上的方块，将其移至底部的7格槽位中，凑齐3个相同图案即可消除。若槽位被填满且无法消除，则游戏失败。请使用generate_image把开始按钮搞得炫酷一点，别的不要用图像素材，用js做就行。另外，一次输出做一件事情，不要一次输出一个以上的代码块"
}]
for typ, chunk in agent.stream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config={"configurable": {"thread_id": 1}},
):
    if typ == "messages":
        print(chunk[0].content, end="")
    elif typ == "values":
        print("\n\n---answer---\n\n", chunk)
