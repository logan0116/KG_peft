# KG peft

## stage_1

bert full fine-tuning vs peft

### result table

| model                 | time for each epoch | GPU Memory | accuracy | f1    | precision | recall |
|-----------------------|---------------------|------------|----------|-------|-----------|--------|
| bert4sc               | 17:42               | 11.866G    | 0.902    | 0.9   | 0.869     | 0.946  |
| bert4sc (freeze bert) | 6:37                | 1.394G     | 0.712    | 0.732 | 0.671     | 0.833  |
| peft (r = 16)         | 15:22               | 9.768G     | 0.918    | 0.914 | 0.911     | 0.927  |
| peft (r = 8)          | 15:10               | 9.758      | 0.918    | 0.913 | 0.912     | 0.925  |
| peft (r = 4)          | 15:24               | 9.752G     | 0.908    | 0.897 | 0.947     | 0.863  |
