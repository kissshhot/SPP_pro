import json
import jsonlines
import os
file='trivia_creative_writing_100_n_5.jsonl__method-spp_engine-devgpt4-32k_temp-0.0_topp-1.0_start0-end100.jsonl'
DATA_PATH = './agent/SPP/logs/trivia_creative_writing/w_sys_mes/'
num_correct = 0
path = os.path.join(DATA_PATH, file)
with open(path, "r") as f:
    data = [json.loads(line) for line in f]
for i in range(0, 100):
    try:
        num_correct += data[i]['test_output_infos'][0]['correct_count']
    except:
        pass
print(num_correct)