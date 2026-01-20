"""
统一的模型配置文件
用于配置 OpenAI Compatible API

使用方法:
1. 在 .env 文件中配置以下环境变量:
   OPENAI_API_KEY='你的API密钥'
   OPENAI_BASE_URL='你的API地址'  # 例如: https://api.example.com/v1
   OPENAI_MODEL_NAME='你的模型名称'  # 例如: gpt-4, qwen-turbo 等

2. 在 notebook 中导入使用:
   from model_config import get_model, get_model_name
   model = get_model()
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 从环境变量读取配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")


def get_model_name() -> str:
    """获取配置的模型名称"""
    return OPENAI_MODEL_NAME


def get_model(**kwargs):
    """
    获取配置好的 ChatModel 实例

    Args:
        **kwargs: 传递给 init_chat_model 的额外参数，如 temperature 等

    Returns:
        配置好的 ChatModel 实例
    """
    from langchain.chat_models import init_chat_model

    model_kwargs = {
        "model": OPENAI_MODEL_NAME,
        "model_provider": "openai",
    }

    # 只有当 base_url 配置了才添加
    if OPENAI_BASE_URL:
        model_kwargs["base_url"] = OPENAI_BASE_URL

    if OPENAI_API_KEY:
        model_kwargs["api_key"] = OPENAI_API_KEY

    # 合并用户传入的额外参数
    model_kwargs.update(kwargs)

    return init_chat_model(**model_kwargs)


def get_agent(**kwargs):
    """
    获取配置好的 Agent 实例

    Args:
        **kwargs: 传递给 create_agent 的额外参数，如 tools, prompt 等

    Returns:
        配置好的 Agent 实例
    """
    from langchain.agents import create_agent

    # 如果用户没有传入 model，使用默认配置
    if "model" not in kwargs:
        kwargs["model"] = get_model()

    return create_agent(**kwargs)
