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


 python3 convert-hf-to-gguf.py /home/mozinode4p/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat/snapshots/08914867436b750c287539795e63c24631273878/ --outfile ../model/model_path.gguf
