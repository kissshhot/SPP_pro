import os
import json
import argparse
from models import OpenAIWrapper
from tasks import get_task
import time
from tqdm import tqdm
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModel
from prompts.role_describe import historyExpert1, musicExpert1, storyWriter1, historyExpert2, historyExpert3, musicExpert2, musicExpert3, storyWriter2, storyWriter3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map = 'auto' if torch.cuda.is_available() else None)
tokenizer = AutoTokenizer.from_pretrained("/data1/dyf/hub/Mistral-7B-Instruct-v0.2_historyExpert_lora_merged/",use_fast='store_true')
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = 32000
from peft import PeftModel
#使用peftmodel加载lora
lora_model_name_or_path = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_storyWriter_lora/"
#### 初始化PeftModel, 并且load第一个adapter
lora_model = PeftModel.from_pretrained(model, model_id = lora_model_name_or_path, adapter_name = "storyWriter")
#### 读取另外两个adapter
lora_model.load_adapter(model_id = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_musicExpert_lora/",adapter_name = "musicExpert")
lora_model.load_adapter(model_id = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_historyExpert_lora/", adapter_name = "historyExpert")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map = 'auto' if torch.cuda.is_available() else None)
SLEEP_RATE = 10 # sleep between calls
def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def _post_process_raw_response(task, raw_output_batch, method):
    unwrapped_output_batch = []
    if_success_batch = []
    for output in raw_output_batch:
        unwrapped_output, if_success_flag = task.prompt_unwrap(output, method)
        unwrapped_output_batch.append(unwrapped_output)
        if_success_batch.append(if_success_flag)
    return unwrapped_output_batch, if_success_batch

def _run_task(task_name, gpt, task, i, method, num_generation):
    if task_name in ['trivia_creative_writing', 'logic_grid_puzzle']:
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
        # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map = 'auto' if torch.cuda.is_available() else None)
        # tokenizer = AutoTokenizer.from_pretrained("/data1/dyf/hub/Mistral-7B-Instruct-v0.2_historyExpert_lora_merged/",use_fast='store_true')
        # model.resize_token_embeddings(len(tokenizer))
        # #使用peftmodel加载lora
        # from peft import PeftModel
        # lora_model_name_or_path = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_storyWriter_lora/"
        # #### 初始化PeftModel, 并且load第一个adapter
        # lora_model = PeftModel.from_pretrained(model, model_id = lora_model_name_or_path, adapter_name = "storyWriter")
        # #### 读取另外两个adapter
        # lora_model.load_adapter(model_id = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_musicExpert_lora/",adapter_name = "musicExpert")
        # lora_model.load_adapter(model_id = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_historyExpert_lora/", adapter_name = "historyExpert")
        # # get prompt
        prompt = task.get_input_prompt(i, method=method)
        system_message = ""
        # get raw response
        #这里改写为通过加载本地模型
        # 第一次generate必定是基础模型
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # overload system message
        messages = [
            {"role":"user","content":prompt}
        ]
        #使用基础模型
        #os.environ["TORCH_USE_CUDA_DSA"] = "1"
        device = "cuda" # the device to load the model onto
        # device_map = {'cpu': 'cpu', 'cuda:0': 'cuda:0', 'cuda:1': 'cuda:1', 'cuda:2': 'cuda:2', 'cuda:3': 'cuda:3'}
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        #model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map = 'auto' if torch.cuda.is_available() else None)
        # tokenizer = AutoTokenizer.from_pretrained("/data1/dyf/hub/Mistral-7B-Instruct-v0.2_historyExpert_lora_merged/",use_fast='store_true')
        # model.resize_token_embeddings(len(tokenizer))
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        # model.to(device)
        with lora_model.disable_adapter():
            generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens=5000, pad_token_id = 32000)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_output_batch = decoded
        # time.sleep(5)

        # #使用peftmodel加载lora
        # lora_model_name_or_path = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_storyWriter_lora/"
        # #### 初始化PeftModel, 并且load第一个adapter
        # lora_model = PeftModel.from_pretrained(model, model_id = lora_model_name_or_path, adapter_name = "storyWriter")
        # #### 读取另外两个adapter
        # lora_model.load_adapter(model_id = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_musicExpert_lora/",adapter_name = "musicExpert")
        # lora_model.load_adapter(model_id = "/data1/dyf/hub/Mistral-7B-Instruct-v0.2_historyExpert_lora/", adapter_name = "historyExpert")
        # return decoded, agent_string
        if raw_output_batch == []: # handle exception
            return {}
        while True:
            # 检查每次输出内容是否重复
            # print(tokenizer.batch_decode(generated_ids)[0])
            #停止的标志是最后四个字符是结束符
            if raw_output_batch[0][-4:] == '</s>':
                break
            if agent_string == 'Assistant' or agent_string == 'final_answer':
                # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,7"
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                #os.environ["TORCH_USE_CUDA_DSA"] = "1"
                model_inputs = torch.load('tensor.pt')
                #将所有的<pad>替换为空格
                # indices = (model_inputs[0] == 32000)
                # model_inputs[0][indices] = 259
                with lora_model.disable_adapter():
                    generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens=5000, pad_token_id = 32000)
                decoded = tokenizer.batch_decode(generated_ids)
                print(decoded[0][-20:])
                raw_output_batch = decoded
                # time.sleep(5) #似乎加上这个效果会好很多
            else:
                #判断文本相似度确定使用哪个agent
                sentences = [agent_string, historyExpert1, historyExpert2, historyExpert3, musicExpert1, musicExpert2, musicExpert3, storyWriter1, storyWriter2, storyWriter3]

                        # Load model from HuggingFace Hub
                tokenizer_text = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
                model_text = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
                model_text.eval()

                # Tokenize sentences
                encoded_input = tokenizer_text(sentences, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    model_output = model_text(**encoded_input)
                    sentence_embeddings = model_output[0][:, 0]
                # normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                score_list = [sentence_embeddings[0] @ sentence_embeddings[1].T, sentence_embeddings[0] @ sentence_embeddings[2].T, sentence_embeddings[0] @ sentence_embeddings[3].T, sentence_embeddings[0] @ sentence_embeddings[4].T, sentence_embeddings[0] @ sentence_embeddings[5].T, sentence_embeddings[0] @ sentence_embeddings[6].T, sentence_embeddings[0] @ sentence_embeddings[7].T, sentence_embeddings[0] @ sentence_embeddings[8].T, sentence_embeddings[0] @ sentence_embeddings[9].T]
                print(score_list)
                #如果最大相似度不大于0.5，就使用基础模型
                if max(score_list) <= 0.5:
                    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,7"
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                    #os.environ["TORCH_USE_CUDA_DSA"] = "1"
                    device = "cuda" # the device to load the model onto
                    # device_map = {'cpu': 'cpu', 'cuda:0': 'cuda:0', 'cuda:1': 'cuda:1', 'cuda:2': 'cuda:2', 'cuda:3': 'cuda:3'}
                    # model.to(device)
                    model_inputs = torch.load('tensor.pt')
                    #将所有的<pad>替换为空格
                    # indices = (model_inputs[0] == 32000)
                    # model_inputs[0][indices] = 259
                    with lora_model.disable_adapter():
                        generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens=5000, pad_token_id = 32000)
                    #time.sleep(5)
                    # decoded = tokenizer.batch_decode(generated_ids)
                    # raw_output_batch = decoded
                else:
                    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,7"
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                    #os.environ["TORCH_USE_CUDA_DSA"] = "1"
                    s = score_list.index(max(score_list))
                    if s in [0,1,2]:
                        model_inputs = torch.load('tensor.pt')
                        lora_model.set_adapter('historyExpert')
                        generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, pad_token_id = 32000)
                        model_inputs = torch.load('tensor.pt')
                        if type(agent_ids) is not str:
                            test = model_inputs[0][:-agent_ids.size(0)]
                            test = test.unsqueeze(0)
                            torch.save(test, 'tensor.pt')
                            agent_string = "Assistant"
                        #lora_model.set_adapter()
                        #time.sleep(5)
                        # decoded = tokenizer.batch_decode(generated_ids)
                        # raw_output_batch = decoded
                    elif s in [6,7,8]:
                        model_inputs = torch.load('tensor.pt')
                        lora_model.set_adapter('storyWriter')
                        generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, pad_token_id = 32000)
                        model_inputs = torch.load('tensor.pt')
                        if type(agent_ids) is not str:
                            test = model_inputs[0][:-agent_ids.size(0)]
                            test = test.unsqueeze(0)
                            torch.save(test, 'tensor.pt')
                            agent_string = "Assistant"
                        #lora_model.set_adapter()
                        #time.sleep(5)
                        # decoded = tokenizer.batch_decode(generated_ids)
                        # raw_output_batch = decoded
                    else:
                        model_inputs = torch.load('tensor.pt')
                        lora_model.set_adapter('musicExpert')
                        generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, pad_token_id = 32000)
                        model_inputs = torch.load('tensor.pt')
                        if type(agent_ids) is not str:
                            test = model_inputs[0][:-agent_ids.size(0)]
                            test = test.unsqueeze(0)
                            torch.save(test, 'tensor.pt')
                            agent_string = "Assistant"
                        #lora_model.set_adapter()
                        #time.sleep(5)
                        # decoded = tokenizer.batch_decode(generated_ids)
                        # raw_output_batch = decoded
        # raw_output_batch = gpt.run(prompt=prompt, n=num_generation)
        # if raw_output_batch == []: # handle exception
        #     return {}    
        # get parsed response, and the success flags (whether or not the parsing is success) (standard prompt always success)
        unwrapped_output_batch, if_success_batch = _post_process_raw_response(task, raw_output_batch, method)
        # compute automatic metric (different for each task), e.g., if the output contains all the answers
        test_output_infos = [task.test_output(i, output) for output in unwrapped_output_batch]
        # log output
        log_output = {
            "idx": i,
            "unwrapped_output": unwrapped_output_batch,
            "parsing_success_flag": if_success_batch,
            "test_output_infos": test_output_infos
        }
    elif task_name == 'codenames_collaborative':
        # get spymaster hint word
        spymaster_prompt = task.get_input_prompt(i, method=method, role='spymaster')
        raw_spymaster_output, raw_response_spymaster = gpt.run(prompt=spymaster_prompt, n=1)
        # raw_spymaster_output, raw_response_spymaster = gpt.run(prompt=spymaster_prompt, n=1, system_message="You are an AI assistant that plays the Spymaster role in Codenames.")
        if raw_spymaster_output == [] or raw_response_spymaster == []: # handle exception
            return {}
        spymaster_output, if_success_batch_spymaster = _post_process_raw_response(task, raw_spymaster_output, method)
        hint_word = spymaster_output[0].replace(".", "").strip()
        print(f"\tidx: {i} | done spymaster, hint word: {hint_word}")
        # sleep before calling guesser
        time.sleep(SLEEP_RATE)
        # get guesser result
        guesser_prompt = task.get_input_prompt(i, method=method, role='guesser', hint_word=hint_word)
        raw_guesser_output, raw_response_batch_guesser = gpt.run(prompt=guesser_prompt, n=num_generation)
        # raw_guesser_output, raw_response_batch_guesser = gpt.run(prompt=guesser_prompt, n=num_generation, system_message="You are an AI assistant that plays the Guesser role in Codenames.")
        if raw_guesser_output == [] or raw_response_batch_guesser == []: # handle exception
            return {}
        guesser_output_batch, if_success_batch_guesser = _post_process_raw_response(task, raw_guesser_output, method)
        # compute automatic metric (different for each task), e.g., if the output contains all the answers
        test_output_infos = [task.test_output(i, output) for output in guesser_output_batch]
        # log output
        log_output = {
            "idx": i,
            "raw_response_spymaster": raw_response_spymaster,
            "raw_response_guesser": raw_response_batch_guesser,
            "spymaster_output": spymaster_output,
            "guesser_output": guesser_output_batch,
            "hint_word": hint_word,
            "parsing_success_flag_spymaster": if_success_batch_spymaster,
            "parsing_success_flag_guesser": if_success_batch_guesser,
            "test_output_infos": test_output_infos
        }
    else:
        raise NotImplementedError(f"task {task_name} not implemented; please choose from ['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative']")
    #一个任务完成了，此时需要把上一次的任务文件清除
    # 文件路径
    file_path = 'agent_string.pkl'
    # 删除文件
    if os.path.exists(file_path):
        os.remove(file_path)
        print("文件已成功删除")
    else:
        print("文件不存在，无法删除")
    file_path = 'agent_tensor.pkl'
    # 删除文件
    if os.path.exists(file_path):
        os.remove(file_path)
        print("文件已成功删除")
    else:
        print("文件不存在，无法删除")
    # log everything else that is related
    log_output.update(args)
    log_output.update({"task_data":task.get_input(i)})
    return log_output

def run(args):
    # get configs
    gpt_config = args['gpt_config']
    task_name = args['task']
    method = args['method']
    start_idx, end_idx = args['task_start_index'], args['task_end_index']
    task_data_file = args['task_data_file']
    num_generation = args['num_generation']

    additional_output_note = args['additional_output_note']
    system_message = args['system_message']
    print(f"setting default system message: {system_message}")
    
    # setup gpt api
    gpt = OpenAIWrapper(config=gpt_config, system_message=system_message)

    # setup log file
    if system_message == "":
        log_file = f"logs/{task_name}/{task_data_file}__method-{method}_engine-{gpt_config['engine']}_temp-{gpt_config['temperature']}_topp-{gpt_config['top_p']}_start{start_idx}-end{end_idx}{additional_output_note}__without_sys_mes.jsonl"
    else:
        log_file = f"logs/{task_name}/{task_data_file}__method-{method}_engine-{gpt_config['engine']}_temp-{gpt_config['temperature']}_topp-{gpt_config['top_p']}_start{start_idx}-end{end_idx}{additional_output_note}__with_sys_mes.jsonl"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # setup task
    task = get_task(task_name, file=task_data_file)
    
    all_logs = []
    print("start running ... log file:", log_file)
    print('len(task)', len(task))
    print()
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    print("total num of instances:", end - start)
    for i in tqdm(range(start, end)):
        log_output = _run_task(task_name, gpt, task, i, method, num_generation)
        all_logs.append(log_output)
        print("\tidx:", i, "done | usage so far:", gpt.compute_gpt_usage())
        # output log at each iteration
        output_log_jsonl(log_file, all_logs)
        # sleep
        time.sleep(SLEEP_RATE)



# TODO: add your custom model config here:
gpt_configs = {
    #Llama-2-13b-chat-hf
    #vicuna-7b-v1.5
    #Mistral-7B-Instruct-v0.2
    "Mistral-7B-Instruct-v0.2": {
        "engine": None,
        "model": "Mistral-7B-Instruct-v0.2",
        #"model": "Llama-2-7b-chat-hf",
        #"model": "Mistral-7B-Instruct-v0.2",
        "temperature": 0.0,
        "max_tokens": 5000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    }
}

default_gpt_config = {
    "engine": None,
    "temperature": 0.0,
    "max_tokens": 5000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

def parse_args():
    model_choices = list(gpt_configs.keys())
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='Mistral-7B-Instruct-v0.2', choices=model_choices)
    args.add_argument('--method', type=str, default='spp', choices=['standard','cot','spp','spp_profile', 'spp_fixed_persona'])
    args.add_argument('--task', type=str, default='trivia_creative_writing', choices=['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative'])
    args.add_argument('--task_data_file', type=str, default='trivia_creative_writing_100_n_5.jsonl')
    args.add_argument('--task_start_index', type=int, default=96)#52 54 72 91 96
    args.add_argument('--task_end_index', type=int, default=100)
    args.add_argument('--num_generation', type=int, default=1)
    args.add_argument('--additional_output_note', type=str, default="")
    args.add_argument('--temperature', type=float, default=0.0)
    args.add_argument('--top_p', type=float, default=1.0)
    args.add_argument('--system_message', type=str, default="You are an AI assistant that helps people find information.")
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = vars(parse_args())
    model_name = args['model']
    
    if model_name in gpt_configs:
        args['gpt_config'] = gpt_configs[model_name] # our configs
    else:
        args['gpt_config'] = default_gpt_config
        args['gpt_config']['engine'] = model_name
    
    # overwrite temperature and top_p
    args['gpt_config']['temperature'] = args['temperature']
    args['gpt_config']['top_p'] = args['top_p']
    print("run args:", args)
    
    run(args)