# KG peft

注意：不能因为结果是一坨答辩就不记录了

# stage_1

## stage_1_SC

bert full fine-tuning vs peft

### result table

result at epoch 0

| model                 | time for each epoch | GPU Memory | accuracy | f1    | precision | recall |
|-----------------------|---------------------|------------|----------|-------|-----------|--------|
| bert4sc               | 17:42               | 11.866G    | 0.902    | 0.9   | 0.869     | 0.946  |
| bert4sc (freeze bert) | 6:37                | 1.394G     | 0.712    | 0.732 | 0.671     | 0.833  |
| peft (r = 16)         | 15:22               | 9.768G     | 0.918    | 0.914 | 0.911     | 0.927  |
| peft (r = 8)          | 15:10               | 9.758G     | 0.918    | 0.913 | 0.912     | 0.925  |
| peft (r = 4)          | 15:24               | 9.752G     | 0.908    | 0.897 | 0.947     | 0.863  |

result at epoch 4

| model                 | time for each epoch | GPU Memory | accuracy | f1    | precision | recall |
|-----------------------|---------------------|------------|----------|-------|-----------|--------|
| bert4sc               |                     |            |          |       |           |        |
| bert4sc (freeze bert) | 6:37                | 1.394G     | 0.783    | 0.778 | 0.763     | 0.817  |
| peft (r = 16)         | 15:22               | 9.768G     | 0.932    | 0.927 | 0.939     | 0.924  |
| peft (r = 8)          | 15:24               | 9.758G     | 0.933    | 0.928 | 0.928     | 0.937  |
| peft (r = 4)          | 15:24               | 9.752G     | 0.932    | 0.927 | 0.933     | 0.931  |

## stage_1_QA

### result table

result at epoch 0

| model                 | accuracy | f1     | precision | recall |
|-----------------------|----------|--------|-----------|--------|
| bert4qa               | 0.685    | 0.718  | 0.747     | 0.752  |
| bert4qa (freeze bert) | 0.0572   | 0.0668 | 0.0668    | 0.125  |
| lora (r = 16)         | 0.463    | 0.496  | 0.526     | 0.554  |
| lora (r = 8)          | 0.41     | 0.442  | 0.464     | 0.508  |
| lora (r = 4)          | 0.414    | 0.445  | 0.474     | 0.513  |
| dora (r = 8)          | 0.443    | 0.475  | 0.502     | 0.542  |

result for all

| model                 | accuracy | f1    | precision | recall | best epoch |
|-----------------------|----------|-------|-----------|--------|------------|
| bert4qa               | 0.696    | 0.729 | 0.771     | 0.753  | 1          |
| bert4qa (freeze bert) | 0.102    | 0.114 | 0.114     | 0.196  | 4          |
| lora (r = 16)         | 0.603    | 0.638 | 0.668     | 0.679  | 4          |
| lora (r = 8)          | 0.608    | 0.643 | 0.677     | 0.684  | 4          |
| lora (r = 4)          | 0.596    | 0.63  | 0.666     | 0.67   | 4          |
| dora (r = 8)          | 0.63     | 0.634 | 0.67      | 0.675  | 4          |

组织我们自己的框架

针对任务：QA
QA的两种形式：

1. 抽取式：question, context, start_pos, end_pos
2. 生成式：question, context, answer

## stage_1_ner

### result table

result at epoch 0

| model                  | f1     |
|------------------------|--------|
| bert4ner               | 0.745  |
| bert4ner (freeze bert) | 0.0428 |
| lora (r = 16)          | 0.0671 |
| lora (r = 8)           | 0.0874 |
| lora (r = 4)           | 0.0728 |

result at epoch 30

| model                  | f1    | best epoch |
|------------------------|-------|------------|
| bert4ner               | 0.795 | 6          |
| bert4ner (freeze bert) | 0.531 | 25         |
| lora (r = 16)          | 0.777 | 27         |
| lora (r = 8)           | 0.771 | 26         |
| lora (r = 4)           | 0.776 | 28         |

# stage_2

~~这一阶段的基础是DBLP知识图谱~~

我们这里的问答数据集和第一阶段的一样还是SQuAD 2.0

后续迭代注意两个问题：

1. 去除重复问题 [done]
2. prompt的设计对答案的影响 [done]

探究微调对于QA任务的影响

0. few-shot [done]
1. 基本的微调 [done]
2. 参考其他的数据集构建的微调策略 [training]
3. 结合知识图谱的微调策略

评价指标的设计

## result table

|          | qwen1.5-gguf-q_4 | qwen1.5-gguf-q_8 | llama3-gguf-q_4 | llama3-gguf-q_8 |
|----------|------------------|------------------|-----------------|-----------------|
| accuracy | 0.96             | 0.95             | 0.944           | 0.935           |
| time     | 3193.28          | 3825.57          | 1074.61         | 1244.64         |

