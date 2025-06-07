import json

import requests
from mcp.server.fastmcp import FastMCP
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(base_url="http://38.246.112.171:3000/", api_key=os.getenv("OPENAI_API_KEY"))


# mcp = FastMCP(
#     "game_stamp",
# )


# @mcp.tool()
def generate_image(description: str, output_dir: str, file_name: str) -> str:
    """
    Generate an image based on a description using GPT-4o Vision
    Example usage: generate_image(description="A beautiful sunset over a calm ocean", output_dir="./output", file_name="sunset.png")

    :param description: The description of the image to generate.
    :param output_dir: The directory to save the generated image.
    :param file_name: The name of the generated image.
    :return: A message indicating the success or failure of the image generation.
    """
    try:
        # Create a prompt that includes the image specifications
        prompt = f"生成游戏素材，png，透明背景。 {description}"
        url = f"{os.getenv('IMAGE_API_BASE')}/v1/chat/completions"

        payload = json.dumps({
            "stream": False,
            "model": "gpt-4o-image-vip",
            "messages": [
                {
                    "content": prompt,
                    "role": "user"
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("IMAGE_API_KEY")}'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        lines = response.text.split('\\n')
        line = ''
        for line in lines:
            if line.startswith('[点击下载]'):
                break
        url = line.replace('[点击下载]', '').strip("()")
        print(f"image url:{url}")
        download_image(url, output_dir, file_name)

        return f"Image successfully generated and saved to {output_dir}"

    except Exception as e:
        return f"Error generating image: {str(e)}"


def download_image(url, save_path, new_name):
    try:
        # 发送 HTTP GET 请求获取图片
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 确保保存路径的目录存在
        os.makedirs(save_path, exist_ok=True)

        # 拼接新的文件路径和名称
        file_extension = os.path.splitext(url)[1]  # 获取原始文件的扩展名
        new_file_path = os.path.join(save_path, f"{new_name}{file_extension}")

        # 写入文件
        with open(new_file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"图片已成功下载并保存为: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"下载图片时出错: {e}")
        return None

# if __name__ == "__main__":
#     mcp.run()
if __name__ == '__main__':
    # download_image("https://filesystem.site/cdn/download/20250607/7VqeOZt2C0DI3QFZy4cYGz9ygPyVoM.png",'game_assets','start_button.png')
    os.makedirs('game_assets', exist_ok=True)
