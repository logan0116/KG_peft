docker run \
  -it \
  --rm \
  --name benchmark \
  --network=host \
  --shm-size 64G \
  --gpus all \
  -v /home/mozinode4p/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinode4p/PycharmProjects/KG_peft/:/Kg_peft/ \
  -w /Kg_peft/quantization_test/ \
  llm:v1.4.1 \
  python3 benchmark_gguf.py --ckpt_dir /Kg_peft/model/qwen1_5-110b-chat-q2_k.gguf --param_size 110b
