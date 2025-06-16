import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT

load_dotenv()

# 检查是否有阿里云API密钥，如果有则使用阿里云，否则使用OpenAI
if os.getenv("ALIYUN_API_KEY"):
    try:
        import dashscope
        from dashscope import Generation
        use_alibaba = True
        dashscope.api_key = os.getenv("ALIYUN_API_KEY")
    except ImportError:
        print("Warning: dashscope not installed, falling back to OpenAI")
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("ALIYUN_API_KEY"))  # 使用阿里云密钥作为fallback
        use_alibaba = False
else:
    from openai import OpenAI
    openai_client = OpenAI()
    use_alibaba = False


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        if use_alibaba:
            # 使用阿里云DashScope
            response = Generation.call(
                model='qwen-plus',
                prompt=f"{MEMORY_CATEGORIZATION_PROMPT}\n\nMemory: {memory}\n\nPlease respond with a JSON object containing a 'categories' array.",
                result_format='message'
            )
            
            if response.status_code == 200:
                content = response.output.text
                # 尝试解析JSON响应
                try:
                    import json
                    if '{' in content and '}' in content:
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        json_content = content[json_start:json_end]
                        categories_data = json.loads(json_content)
                        if 'categories' in categories_data:
                            return categories_data['categories']
                except:
                    pass
                # 如果JSON解析失败，返回默认分类
                return ["general"]
            else:
                print(f"Alibaba API error: {response.message}")
                return ["general"]
        else:
            # 使用OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
                    {"role": "user", "content": memory}
                ],
                response_format={"type": "json_object"}
            )
            
            categories_data = MemoryCategories.model_validate_json(response.choices[0].message.content)
            return categories_data.categories
    except Exception as e:
        print(f"Error in categorization: {e}")
        return ["general"]
