for part in 0
do
  docker run \
    -d \
    --rm \
    --name benchmark_$part \
    --network=host \
    --shm-size 32G \
    --gpus "device=0" \
    -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
    -v /home/mozinodej/PycharmProjects/Kg_peft/:/Kg_peft/ \
    -w /Kg_peft/stage_2_qa/ \
    llm:v1.4 \
    python3 main.py --part $part
done


for part in 1
do
  docker run \
    -d \
    --rm \
    --name benchmark_$part \
    --network=host \
    --shm-size 32G \
    --gpus "device=1" \
    -v /home/mozinodej/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
    -v /home/mozinodej/PycharmProjects/Kg_peft/:/Kg_peft/ \
    -w /Kg_peft/stage_2_qa/ \
    llm:v1.4 \
    python3 main.py --part $part
done