import json
import multiprocessing
import time
import random
from utils import retry_query_gptv2



class MultiProcessingQuery:

  gpt_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "gpt-4o", "gpt-4o-mini"]
  claude_models = ["claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-sonnet-20240229"]
  
  def __init__(self, output_file, model="claude-3-haiku-20240307", is_json=False):
    '''
    model:
      claude-3-haiku-20240307
      claude-3-opus-20240229
      claude-3-sonnet-20240229
      gpt-3.5-turbo
      gpt-3.5-turbo-16k
      gpt-4
      gpt-4-32k
    is_json: whether to save the results in json format
    '''
    self.model = model
    self.output_file = output_file
    self.is_json = is_json
    self.json_retry = 3

  def read_from_json(self, file_path):
    results = {}
    try:
      with open(file_path, 'r') as file:
        for line in file:
          result = json.loads(line.strip())
          results.update(result)
    except FileNotFoundError:
      print("The file was not found.")
    except json.JSONDecodeError:
      print("Error decoding JSON.")
    return results

  def query_llm(self, prompt, schema=None):
    try:
      if self.model in self.gpt_models:
        answer = retry_query_gptv2(prompt, model_name=self.model)
    #   elif self.model in self.claude_models:
    #     answer = Tools.query_claude(prompt, model_name=self.model)
    #   else:
    #     answer = Tools.query_llm(prompt, model_name=self.model)
      return answer
    except Exception as e:
      raise e

  def save_to_json(self, key, result):
    with open(self.output_file, "a") as f:
      json.dump({key: result}, f)
      f.write("\n")  # 为了更好的可读性，每个结果单独一行

  def process_func(self, key_value):
    key, value_pair = key_value
    if isinstance(value_pair, tuple):
      value, schema = value_pair
    else:
      value = value_pair
      schema = None
    retry = 0
    err = None
    while retry < self.json_retry:
      try:
        res = self.query_llm(value, schema)
        
        if self.is_json and isinstance(res, str):
          res = res.strip()
          if res.startswith('```json\n'):
            res = res[8:]
          if res.endswith('```'):
            res = res[:-3]
          # just in case the result is not a json format
          try:
            _ = json.loads(res)
          except json.JSONDecodeError as e:
            retry += 1
            err = e
            continue
        
        result = res
        break
      except Exception as e:
        err = e
        print(f"Unexpected Error processing {key=}: {err}")
        result = None
        break
    else:
      print(f"Error processing {key=}: {err}")
      result = None
    
    self.save_to_json(key, result)

  def query_all_dicts(self, dict_questions, worker_num=4):
    pool = multiprocessing.Pool(processes=worker_num)
    try:
      with pool:
        pool.map(self.process_func, dict_questions.items())
    except KeyboardInterrupt:
      pool.close()
      pool.terminate()
      pool.join()
      exit(1) # still need to exit the program by ps -a and kill -9

  def convert_list_to_dict(self, list_questions, list_schema=None):
    if not list_schema:
      return {str(i): question for i, question in enumerate(list_questions)}
    else:
      # list_schema is for generating strict json schema, each schema in list_schema corresponds to each question in list_questions
      return {str(i): (question, list_schema[i]) for i, question in enumerate(list_questions)}

  def convert_json_to_dict(self, json_file):
    results = self.read_from_json(json_file)
    result_list = [0 for _ in range(len(list(results.values())))]
    for key, value in results.items():
      result_list[int(key)] = value
    return result_list

  def query_a_list(self, list_questions, worker_num=4, list_schema=None):
    dict_questions = self.convert_list_to_dict(list_questions, list_schema)
    self.query_all_dicts(dict_questions, worker_num)
    return self.convert_json_to_dict(self.output_file)


if __name__ == "__main__":
  sample_dict = {
      "key1": "What is the capital of France?",
      "key2": "Where is Beijing",
      "key3": "how are you?",
      "key4": "good morning",
      "key5": "good evening",
  }

  llm = MultiProcessingQuery('test.json')
  llm.query_all_dicts(sample_dict)
