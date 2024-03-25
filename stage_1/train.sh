docker run \
  -it \
  --rm \
  --name peft_starg_2 \
  --network=host \
  --shm-size 32G \
  --gpus all \
  -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinodej/PycharmProjects/Kg_peft/:/Kg_peft/ \
  -w /Kg_peft/stage_1/ \
  llm:v1.3 \
  python3 main.py