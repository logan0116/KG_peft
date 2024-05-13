  docker run \
    -it \
    --rm \
    --network=host \
    --name chat_server \
    --gpus all \
    --shm-size 64G \
    -v /home/mozinode4p/PycharmProjects/KG_peft/chat_server:/chat_server \
    -w /chat_server \
    llm:v1.4 \
    python3 main.py --port 9010