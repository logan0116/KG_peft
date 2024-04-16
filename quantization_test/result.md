# quantization

## 1. Introduction

对比的对象 Qwen

量化方法：

* gptq
    * Qwen1.5-7B-Chat-4bit-128gr-no-desc-act
    * Qwen1.5-7B-Chat-4bit-128gr-desc-act
* awq
    * Qwen1.5-7B-Chat-awq

result table

| model | gptq-no-desc-act | gptq-desc-act | awq     |
|-------|------------------|---------------|---------|
| ACC   | 0.5580           | 0.5912        | 0.5601  |
| time  | 3741.81          | 3745.25       | 5181.86 |

gguf 4bit

| model | gguf | gptq-no-desc-act-gguf | gptq-desc-act-gguf | awq-gguf |
|-------|------|-----------------------|--------------------|----------|