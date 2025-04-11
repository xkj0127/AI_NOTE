from openai import OpenAI
import os
import base64


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 将xxxx/test.png替换为你本地图像的绝对路径
base64_image = encode_image(r"D:\Python_workspace\AI_NOTE\xxx\xxx.png")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key='xx',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
                {"type": "text", "text": '''
                你是一个笔记图像理解助手，图片表达了什么? 请遵循以下指南：
                - 不要解释任何图中文字的概念
                - 用最简练的话告诉我图像的主要内容
                - 如果是图表请告诉我表的数据
                - 告诉我图片类型就不要继续解释这类图片的特点了，例如：知识图谱，照片，柱状图，表格等


                '''},
            ],
        }
    ],
)
print(completion.choices[0].message.content)