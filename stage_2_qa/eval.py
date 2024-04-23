import json
import requests
import logging
import time

# logging config
logging.basicConfig(level=logging.INFO,
                    format="Logan233: %(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def qa_eval_llm(model_type):
    """
    借助大语言模型对QA结果进行评估

    大模型 post api:
        http://192.168.1.114:9010/api/smart_qa/chat


    history = {
        "role": "system",
        "content": "In a QA task, we predict an answer based on the "Question" and "Content". "+
        "Please determine whether the "Answer" matches the "Label". "+
        "If the answer matches the label, respond with "True". If it does not match, respond with "False"."+
        "The match does not require identical wording; rather, the content must convey the same meaning. "
    }

    prompt = "Question: "+question+"\nContent: "+context+"\nLabel: "+label+"\nAnswer: "+answer

    request body:
        {
            "inputs": prompt,
            "history": history
        }

    response body:
        {
            "code": 200,
            "msg": "success",
            "data": output
        }

    """
    # load data
    with open('data/qa_{}_result.json'.format(model_type), 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # history
    history = {
        "role": "system",
        "content": 'In a QA task, we predict an answer based on the "Question" and "Content". ' +
                   'Please determine whether the "Answer" matches the "Label". ' +
                   'If the answer matches the label, respond with "True". If it does not match, respond with "False".' +
                   'The match does not require identical wording; rather, the content must convey the same meaning. '
    }

    # calculate accuracy
    correct_count = 0
    # start time
    logging.info("Start evaluating...")
    time_start = time.time()
    for data in data_list:
        prompt = ("Question: " + data['question'] +
                  "\nContent: " + data['context'] +
                  "\nLabel: " + data['label'] +
                  "\nAnswer: " + data['answer'])
        req = {
            "inputs": prompt,
            "history": history
        }

        # post api
        response = requests.post("http://192.168.1.114:9010/api/smart_qa/chat", json=req)
        output = response.json()['data']

        if output == "True":
            correct_count += 1

    # end time
    time_end = time.time()

    accuracy = correct_count / len(data_list)
    logging.info("Accuracy: %s" % accuracy)
    logging.info("Time: %s" % (time_end - time_start))


if __name__ == '__main__':
    qa_eval_llm("gguf")
