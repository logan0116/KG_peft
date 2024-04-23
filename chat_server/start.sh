  docker run \
    -it \
    --rm \
    --network=host \
    --name chat_server \
    --gpus all \
    --shm-size 32G \
    -v /root/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ \
    -v /home/python_projects/Kg_peft:/Kg_peft \
    -w /Kg_peft/chat_server \
    llm:v1.4 \
    python3 main.py --port 9010