# KG peft

## stage_1

bert full fine-tuning vs peft

### result table

| model                 | time for each epoch | accuracy | f1    | precision | recall |
|-----------------------|---------------------|----------|-------|-----------|--------|
| bert4sc               | 17:42               | 0.902    | 0.9   | 0.869     | 0.946  |
| bert4sc (freeze bert) | 6:37                | 0.712    | 0.732 | 0.671     | 0.833  |
| peft                  |                     |          |       |           |        |