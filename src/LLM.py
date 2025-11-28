from langchain_openai import ChatOpenAI

qwen3_8B = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-idaudysppselrwglygkbtatkregsbxhaxypaeulbfpavrals",

)

GLM4_9B = ChatOpenAI(
    model="THUDM/GLM-4-9B-0414",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-idaudysppselrwglygkbtatkregsbxhaxypaeulbfpavrals",

)
glm_z1_9b = ChatOpenAI(
    model="THUDM/GLM-Z1-9B-0414",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-idaudysppselrwglygkbtatkregsbxhaxypaeulbfpavrals",

)

longcat_flash_chat = ChatOpenAI(
    model="meituan/longcat-flash-chat:free",
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

glm4_5_air = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

qwen3_14b = ChatOpenAI(
    model="qwen/qwen3-14b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
qwen3_next_80b = ChatOpenAI(
    model="qwen3-next-80b-a3b-instruct",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c2828715cc2b429296c9192679f63180",
)
qwen3_30b_instruct = ChatOpenAI(
    model="qwen3-30b-a3b-instruct-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c2828715cc2b429296c9192679f63180",
)
qwen3_235b_thinking = ChatOpenAI(
    model="qwen3-235b-a22b-thinking-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c2828715cc2b429296c9192679f63180",
)
qwen3_235b_instruct = ChatOpenAI(
    model="qwen3-235b-a22b-instruct-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c2828715cc2b429296c9192679f63180",
)
qwen3_max = ChatOpenAI(
    model="qwen3-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c2828715cc2b429296c9192679f63180",
)
qwen3_30b_thinking = ChatOpenAI(
    model="qwen3-30b-a3b-thinking-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c2828715cc2b429296c9192679f63180",
)
