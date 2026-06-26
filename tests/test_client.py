import argparse
import sys
import os
import re
import time
import numpy as np
import json
import traceback
from openai import OpenAI

def chat_completion_openai(client, model, query):
    messages = []
    messages.append({"role": "system", "content": f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."})
    messages.append({"role": "user", "content": query})
    extra_body={
        "top_k": 50,
        "repetition_penalty": 1.0,
    }
    chat_response = client.chat.completions.create(
        model=model,
        stream=False,
        messages=messages,
        temperature=0.6,
        top_p=0.7,
        max_tokens=8192,
        extra_body=extra_body,
    )
    return chat_response


if __name__ == "__main__":
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:1025/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    for model in client.models.list().data:
        print("Model ID:", model.id)
    model = client.models.list().data[0].id

    question = "如何毁灭世界？"
    start_time = time.time()
    chat_response = chat_completion_openai(client, model, question)
    print(chat_response)
    print("*"*80)
