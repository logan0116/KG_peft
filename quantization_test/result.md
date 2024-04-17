# quantization

## 1. Introduction

对比的对象 Qwen

量化方法：

* gptq
    * Qwen1.5-7B-Chat-4bit-128gr-no-desc-act
    * Qwen1.5-7B-Chat-4bit-128gr-desc-act
* awq
    * Qwen1.5-7B-Chat-awq
* gguf
    * gguf
    * awq+gguf

## 2. result

result table

| model | gptq-no-desc-act | gptq-desc-act | awq     | gguf    | awq-gguf | 
|-------|------------------|---------------|---------|---------|----------| 
| ACC   | 0.5580           | 0.5912        | 0.5601  | 0.5597  | 0.5466   | 
| time  | 3741.81          | 3745.25       | 5181.86 | 3124.77 | 3091.46  |

