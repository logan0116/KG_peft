docker run \
  -it \
  --rm \
  --name quantization \
  --network=host \
  --shm-size 32G \
  --gpus "device=0" \
  -v //home/mozinode4p/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinode4p/PycharmProjects/Kg_peft/:/Kg_peft/ \
  -w /Kg_peft/quantization_test/ \
  llm:v1.4.1