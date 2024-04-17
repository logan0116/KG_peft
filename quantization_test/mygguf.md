# convert

```Bash
# Path: llama.cpp
python3 convert-hf-to-gguf.py model_path --outfile model_path.gguf
```

# quantize

```Bash
# Path: Kg_peft/quantization_test
# quantize
/llama.cpp/build/bin/quantize Qwen1.5-7B-Chat.gguf Qwen1.5-7B-Chat-q4_0.gguf q4_0
```


