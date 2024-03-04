import json
import jsonlines
import os
file1='trivia_creative_writing_100_n_4.jsonl__method-spp_engine-None_temp-0.0_topp-1.0_start0-end100__with_sys_mes.jsonl'
file_llama7b = "log备份/trivia_creative_writing_100_n_5.jsonl__method-spp_temp-0.0_topp-1.0_start0-end10__with_sys_mes_llama7b.jsonl"
file_lora = "log备份/trivia_creative_writing_100_n_5.jsonl__method-spp_temp-0.0_topp-1.0_start0-end10__with_sys_mes_lora.jsonl"
file_mis = "log备份/trivia_creative_writing_100_n_5.jsonl__method-spp_temp-0.0_topp-1.0_start0-end10__with_sys_mes.jsonl"
file_no_llama7b = "agent/SPP/logs/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl__method-spp_engine-None_temp-0.0_topp-1.0_start0-end10__with_sys_mes_llama7b.jsonl"
file_no_mistral = "agent/SPP/logs/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl__method-spp_engine-None_temp-0.0_topp-1.0_start0-end10__with_sys_mes.jsonl"
file_no_mistral_8bit = "agent/SPP/logs/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl__method-spp_engine-None_temp-0.0_topp-1.0_start0-end10__with_sys_mes_mistral_8bit.jsonl"
file_no_mistral_8bit_10_100 = "agent/SPP/logs/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl__method-spp_engine-None_temp-0.0_topp-1.0_start10-end100__with_sys_mes.jsonl"
num_correct = 0
# path = os.path.join(DATA_PATH3, file1)
with open(file_no_mistral_8bit_10_100, "r") as f:
    data = [json.loads(line) for line in f]
for i in range(0, 100):
    try:
        num_correct += data[i]['test_output_infos'][0]['correct_count']
    except:
        pass
print(f'mistral-7b:{num_correct}/50')
num_correct = 0






# num_correct = 0
# path = os.path.join(DATA_PATH1, file1)
# with open(path, "r") as f:
#     data = [json.loads(line) for line in f]
# for i in range(0, 100):
#     try:
#         num_correct += data[i]['test_output_infos'][0]['correct_count']
#     except:
#         pass
# print('llama2-7b:{num_correct}/100')
