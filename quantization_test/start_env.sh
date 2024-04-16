docker run \
  -it \
  --rm \
  --name quantization \
  --network=host \
  --shm-size 32G \
  --gpus "device=0" \
  -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinodej/.cache/huggingface/datasets/:/root/.cache/huggingface/datasets/ \
  -v /home/mozinodej/PycharmProjects/Kg_peft/:/Kg_peft/ \
  -w /Kg_peft/quantization_test/ \
  llm:v1.3.4