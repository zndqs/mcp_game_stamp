import replicate
from dotenv import load_dotenv
import os

from langchain_core.tools import tool
import pathlib

load_dotenv()


def generate_image(description: str, output_dir: str, file_name: str, aspect_ratio: str) -> str:
    """
    Generate a svg image based on a description using recraft-ai / recraft-v3-svg

    Example usage:
        generate_image(description="A beautiful sunset over a calm ocean", output_dir="./output", file_name="sunset")

    Note: use SVG as follows; otherwise, a white background will appear instead of a transparent one
        ```html
        <embed src="./output/sunset.svg" width="300" height="100"
        type="image/svg+xml"
        pluginspage="http://www.adobe.com/svg/viewer/install/" />
        ```

    :param description: The description of the image to generate.
    :param output_dir: The directory to save the generated image.
    :param file_name: The name of the generated image.
    :param aspect_ratio: The aspect ratio of the generated image.
    :return: A message indicating the success or failure of the image generation.
    """
    try:
        input = {
            "prompt": f"生成一个SVG格式的矢量图形，扁平化设计，简洁的线条，有限的颜色（不超过5种），清晰的轮廓，无渐变和阴影。图形描述：{description}。该图形将用于游戏素材。",
            "aspect_ratio": aspect_ratio
        }
        output = replicate.run(
            "recraft-ai/recraft-v3-svg",
            input=input
        )
        # 检查输出目录是否存在，如果不存在则创建
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path.joinpath(f"{file_name}.svg"), "wb") as file:
            file.write(output.read())
            print(f"generate finish :{description}")
            return f"Image successfully generated and saved to {os.path.join(output_dir, f'{file_name}.svg')}"

    except Exception as e:
        return f"Error generating image: {str(e)}"


generate_image_tool = tool()(generate_image)
