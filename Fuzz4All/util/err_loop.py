from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from Fuzz4All.target.target import Target

def read_file_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"


def write_string_to_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def extract_content(text, start_keyword, end_keyword):
    # 构建正则表达式
    pattern = re.compile(re.escape(start_keyword) + r'(.*?)' + re.escape(end_keyword), re.DOTALL)
    # 搜索匹配的内容
    match = pattern.search(text)
    if match:
        # 返回匹配的内容
        return match.group(1)
    else:
        return None


def extract_generate_code(generate_string: str):
    #start_keyword = "Here is the corrected code:\n\n```"
    start_keyword = "```"
    end_keyword = "```"  
    extracted_content = extract_content(generate_string, start_keyword, end_keyword)
    if extracted_content:
        print("提取的内容:")
        print(extracted_content)
    else:
        print("未找到匹配的内容。")
        extracted_content = "none"
    return extracted_content



def err_fix(model, tokenizer, file_name, message: str):
    #tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    
    code = read_file_to_string(file_name)
    user_content = code + '\n<code2err>\n' + message
    user_content = user_content[0:min(len(user_content), 6000)]
    print("00000000000000000000000000000000000000000000")
    print(user_content)
    print("00000000000000000000000000000000000000000000")

    messages = [
        {"role": "system", 
        "content": 
            '''
            Your task now is to fix the bug in the code.
            Code and its error are separated by token <code2err>.
            '''
        },
        {"role": "user", "content":user_content},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # tokenizer.eos_token_id is the id of <|EOT|> token
    outputs = model.generate(inputs, max_new_tokens=1024, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

    print("222222222222222222222222222222222222222222222222222222222222222222222")
    print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
    print("222222222222222222222222222222222222222222222222222222222222222222222")

    out_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    out_text = extract_generate_code(out_text)
    if out_text == "none":
        write_string_to_file(file_name, code)
        return
    # 对out_text格式进行清洗，删除最开头的```python
    for x in out_text:
        if x == '\n':
            print(x)
            break
        out_text = out_text.lstrip(x)

    print("11111111111111111111111111111111111111111111111111111")
    print(out_text)
    print("11111111111111111111111111111111111111111111111111111")
    
    #把大模型修改后的代码写回到.fuzz文件中
    write_string_to_file(file_name, out_text)
    return out_text


