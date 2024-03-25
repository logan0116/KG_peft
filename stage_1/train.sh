docker run \
  -it \
  --rm \
  --name peft_starg_1 \
  --network=host \
  --shm-size 32G \
  --gpus all \
  -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinodej/PycharmProjects/peft_test/:/peft_test/ \
  -w /peft_test/stage_1/ \
  llm:v1.3 \
  python3 main.py