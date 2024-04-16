docker run \
  -it \
  --rm \
  --name benchmark \
  --network=host \
  --shm-size 32G \
  --gpus "device=0" \
  -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinodej/PycharmProjects/Kg_peft/:/Kg_peft/ \
  -w /Kg_peft/quantization_test/ \
  llm:v1.3.2 \
  python3 benchmark.py --ckpt_dir Qwen/Qwen1.5-7B-Chat --param_size 7 --model_type Other
