docker run \
  -it \
  --rm \
  --name benchmark_0 \
  --network=host \
  --shm-size 32G \
  --gpus all \
  -v /home/mozinode4p/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinode4p/PycharmProjects/LLaMA-Factory:/LLaMA-Factory \
  -v /home/mozinode4p/PycharmProjects/KG_peft:/Kg_peft \
  -w /Kg_peft/stage_2_qa/ \
  llm:v1.4.1

