# quantization

## 1. Introduction

对比的对象 Qwen

量化方法：

* gptq
    * Qwen1.5-7B-Chat-4bit-128gr-no-desc-act
    * Qwen1.5-7B-Chat-4bit-128gr-desc-act
* awq
    * Qwen1.5-7B-Chat-awq
* gguf
    * gguf
    * awq+gguf

## 2. result

result table

| model | gptq-no-desc-act | gptq-desc-act | awq     | gguf    | awq-gguf | 
|-------|------------------|---------------|---------|---------|----------| 
| MMLU  | 0.5580           | 0.5912        | 0.5601  | 0.5597  | 0.5466   | 
| time  | 3741.81          | 3745.25       | 5181.86 | 3124.77 | 3091.46  |

# 这算是一个小插曲


ACC-abstract_algebra: 0.4700
ACC-anatomy: 0.7407
ACC-astronomy: 0.8618
ACC-business_ethics: 0.7300
ACC-clinical_knowledge: 0.7132
ACC-college_biology: 0.8958
ACC-college_chemistry: 0.5400
ACC-college_computer_science: 0.6300
ACC-college_mathematics: 0.4400
ACC-college_medicine: 0.7168
ACC-college_physics: 0.5196
ACC-computer_security: 0.8100
ACC-conceptual_physics: 0.7191
ACC-econometrics: 0.5439
ACC-electrical_engineering: 0.7379
ACC-elementary_mathematics: 0.6614
ACC-formal_logic: 0.5397
ACC-global_facts: 0.5500
ACC-high_school_biology: 0.8806
ACC-high_school_chemistry: 0.6305
ACC-high_school_computer_science: 0.7400
ACC-high_school_european_history: 0.0182
ACC-high_school_geography: 0.9091
ACC-high_school_government_and_politics: 0.9793
ACC-high_school_macroeconomics: 0.7897
ACC-high_school_mathematics: 0.4667
ACC-high_school_microeconomics: 0.8277
ACC-high_school_physics: 0.4901
ACC-high_school_psychology: 0.9174
ACC-high_school_statistics: 0.5139
ACC-high_school_us_history: 0.1176
ACC-high_school_world_history: 0.0759
ACC-human_aging: 0.7803
ACC-human_sexuality: 0.8015
ACC-international_law: 0.8512
ACC-jurisprudence: 0.8056
ACC-logical_fallacies: 0.8221
ACC-machine_learning: 0.3929
ACC-management: 0.8641
ACC-marketing: 0.9316
ACC-medical_genetics: 0.8400
ACC-miscellaneous: 0.9017
ACC-moral_disputes: 0.8092
ACC-moral_scenarios: 0.5866
ACC-nutrition: 0.7908
ACC-philosophy: 0.8039
ACC-prehistory: 0.8549
ACC-professional_accounting: 0.5993
ACC-professional_law: 0.5900
ACC-professional_medicine: 0.3713
ACC-professional_psychology: 0.7745
ACC-public_relations: 0.7182
ACC-security_studies: 0.7959
ACC-sociology: 0.8706
ACC-us_foreign_policy: 0.9100
ACC-virology: 0.5843
ACC-world_religions: 0.8538
ACC-all: 0.6946
total run time 26609.67
