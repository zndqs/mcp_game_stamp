import inspect
import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph_codeact import create_codeact
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool as create_tool

from src.game_stamp.server import generate_image_tool

import builtins
import contextlib
import io
from typing import Any, Optional

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

def create_default_prompt(tools: list[StructuredTool], base_prompt: Optional[str] = None):
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You are a professional HTML5 full-stack game development expert capable of generating directly runnable web games based on user requirements. You are proficient in the HTML/CSS/JavaScript trio, and well-versed in core technologies such as Canvas rendering, game loops, and event handling.

You will be given a task to perform. You should output either
- ONE Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in tools:
        prompt += f'''
def {tool.name}{str(inspect.signature(tool.func))}:
    """{tool.description}"""
    ...
'''

    prompt += """

Variables defined at the top level of previous code snippets can be referenced in your code.

Reminder: 
-one step, one code snippet
-use Python code snippets to call tools
-execute actions step by step, pausing after each step to verify success and check if subsequent actions depend on previous results before proceeding. Always confirm dependencies and outcomes before continuing. 
-proper use of the aforementioned tools can enhance the visual appeal and fun of the game, while maintaining a consistent style."""
    return prompt


tools = [generate_image_tool]
model = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="anthropic/claude-sonnet-4",
)

code_act = create_codeact(model, tools, eval,prompt=create_default_prompt(tools))
agent = code_act.compile(checkpointer=MemorySaver())

messages = [{
    "role": "user",
    "content": "生成一个html游戏，屏幕上有很多堆叠非常多层的方块，玩家需要点击屏幕上的方块，将其移至底部的7格槽位中，凑齐3个相同图案即可消除。若槽位被填满且无法消除，则游戏失败。"
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
