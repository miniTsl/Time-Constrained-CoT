import json
import multiprocessing
import time
import random
import ast
from typing import Iterable, Union, Any

def load_jsonl(file) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def dump_jsonl(file, data: Iterable[Any]):
    with open(file, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def dump_npy(file, data):
    import numpy as np
    np.save(file, data)

def load_npy(file):
    import numpy as np
    return np.load(file)

def save_pth(file, data):
    import torch
    torch.save(data, file)

def load_pth(file):
    import torch
    return torch.load(file)

def retry_query_gptv2(prompt: str, model_name: str='gpt-3.5-turbo-16k', retry_times=12):
    from openai import OpenAI
    client = OpenAI(
        base_url='https://api.apikey.vip/v1',
        # This is the default and can be omitted
        api_key='sk-WKwPNrsrZZs7GoeIF0fTklpo4WxlDsKF2U75GuVSVXCQWyhs'
    )
    retry = 0
    while retry < retry_times:
        try:
            completion = client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                timeout=15
            )
            print('gpt answer:', completion)
            res = completion.choices[0].message.content
            break
        except:
            retry += 1
            # pdb.set_trace()
            print(f'\n\n\nWARNING: API failed {retry} times. Retrying...\n\n\n')
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    return res

def _convert_str_to_json(input_str):
    try: 
       converted_answer = ast.literal_eval(input_str)
       return converted_answer
    except:
       converted_answer = json.loads(input_str)
       return converted_answer
    
def convert_gpt_answer_to_json(answer, model_name, default_value={'default': 'format wrong'}, query_func=retry_query_gptv2, strict_json=False, schema=None):
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, list):
        return answer
    import ast
    convert_prompt = f'''
Convert the following data into JSON dict format. Return only the dict. Ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).

data:
{answer}

**Please do not output any content other than the JSON dict format.**
'''
    try:
        answer = answer.replace('```json', '').replace('```dict', '').replace('```list', '').replace('```python', '')
        answer = answer.replace('```', '').strip()

        # converted_answer = ast.literal_eval(answer)
        converted_answer = _convert_str_to_json(answer)

    except:
        print('*'*10, 'converting', '*'*10, '\n', answer, '\n', '*'*50)
        if not strict_json:
            converted_answer = query_func(convert_prompt, model_name)
        else:
            converted_answer = query_func(convert_prompt, model_name, strict_json=True, properties=schema)
        print('*'*10, 'converted v1', '*'*10, '\n', converted_answer, '\n', '*'*10)
        if isinstance(converted_answer, str):
            try:
                converted_answer = converted_answer.replace('```json', '').replace('```dict', '').replace('```list', '').replace('```python', '')
                converted_answer = converted_answer.replace('```', '').strip()
                # converted_answer = ast.literal_eval(converted_answer)
                converted_answer = _convert_str_to_json(converted_answer)
            except:
                new_convert = f'''
Convert the following data into JSON dict format. Return only the JSON dict. Ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).
data:
{answer}

The former answer you returned:
{converted_answer}
is wrong and can not be parsed in python. Please check it and convert it properly!

**Please do not output any content other than the JSON dict format!!!**
'''
                if not strict_json:
                    converted_answer = query_func(new_convert, model_name)
                else:
                    converted_answer = query_func(new_convert, model_name, strict_json=True, properties=schema)
                print('*'*10, 'converted v2', '*'*10, '\n', converted_answer, '\n', '*'*10)
                if isinstance(converted_answer, str):
                    try:
                        converted_answer = converted_answer.replace('```json', '').replace('```dict', '').replace('```list', '').replace('```python', '')
                        converted_answer = converted_answer.replace('```', '').strip()
                        # converted_answer = ast.literal_eval(converted_answer)
                        converted_answer = _convert_str_to_json(converted_answer)
                    except:
                        return default_value
    return converted_answer