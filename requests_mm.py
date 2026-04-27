import requests
import base64


# 读取图片文件并转换为Base64编码
with open('/home/yf/test_mm2.jpg', 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# 构造请求数据
data = {
    "model": "qwen2_5_vl",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the content of this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ],
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "top_k": 64,
    "top_p": 0.95,
    "max_tokens": 8192,
    "stream": False
}

# 设置请求头
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

# 发送POST请求
url = "http://141.61.33.144:9091/v1/chat/completions"
response = requests.post(url, headers=headers, json=data)

# 处理响应
print("Status Code:", response.status_code)
print(response) ## 对象
response_json = response.json()
print("Response:", response.json())
print(response_json["choices"][0]["message"]["content"])