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

result at epoch 5

| model                 | time for each epoch | GPU Memory | accuracy | f1    | precision | recall |
|-----------------------|---------------------|------------|----------|-------|-----------|--------|
| bert4sc               |                     |            |          |       |           |        |
| bert4sc (freeze bert) | 6:37                | 1.394G     | 0.783    | 0.778 | 0.763     | 0.817  |
| peft (r = 16)         | 15:22               | 9.768G     | 0.932    | 0.927 | 0.939     | 0.924  |
| peft (r = 8)          | 15:24               | 9.758G     | 0.933    | 0.928 | 0.928     | 0.937  |
| peft (r = 4)          | 15:24               | 9.752G     | 0.932    | 0.927 | 0.933     | 0.931  |


