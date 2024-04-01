# KG peft

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
    1. classical version: question, context, start_pos, end_pos
    2. LLM version: question, context, answer


