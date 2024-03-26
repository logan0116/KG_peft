docker run \
  -it \
  --rm \
  --name peft_starg_1 \
  --network=host \
  --shm-size 32G \
  --gpus all \
  -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
  -v /home/mozinodej/PycharmProjects/Kg_peft/:/Kg_peft/ \
  -w /Kg_peft/stage_1/ \
  llm:v1.3 \
  python3 main.py --gpu_id 1 --batch_size 16 --epochs 5 --freeze False --peft True --LoRA_r 16 && \
  python3 main.py --gpu_id 1 --batch_size 16 --epochs 5 --freeze False --peft True --LoRA_r 8