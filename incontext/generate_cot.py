import json
from parallel_query import MultiProcessingQuery
from utils import load_jsonl




def make_query_coarse2fine_cot_prompt(task, solution, answer):
    prompt = f'''<|im_start|>system\nSolve the task through two types of reasoning:\n1. Coarse-Grained Reasoning: give quick analysis step by step and an answer. Focus on efficiency and simplicity, and you should be as concise as possible, \n2. Fine-Grained Reasoning: give detailed analysis step by step and a refined answer. Foucus on accuracy and correctness.\nPut final answer within \\boxed{{}}.\n\nOutput format:\n**Coarse Reasoning**\n\n**Fine Reasoning**\n\n**Final Answer** within \\boxed{{}}<|im_end|>

For example: 
//Start of the example
Question: Captain Zarnin of Planet Hvan has four job openings for his battle station: Assistant Engineer, Weapons Maintenance, Field Technician, and Radio Specialist. Zarnin has received 24 resumes and found himself less than thrilled about half of them and does not hire them. The rest, he thinks, could fill any of the open posts. In how many ways can Zarnin staff his battle station?\n\nNote: A candidate can fill at most one job opening.

Answer: "**Coarse Reasoning**\n12 of 24 applicants suitable, \( 12 \times 11 \times 10 \times 9 = 11,880 \) ways. \n\n**Fine Reasoning**\nOf the 24 people who applied, only 12 people are suitable to be hired. Therefore, there are 12 suited to be Assistant Engineer. After that position is filled, there are only 11 left for Weapons Maintenance, then 10 left for Field Technician, and then 9 for Radio Specialist. Therefore, there are $12 \\cdot 11 \\cdot 10 \\cdot 9 = 11,880$ possible ways in which Zarnin can fill his job openings. \n\n**Final Answer**\n\\boxed{{11,880}}"
//End of the example

Now please solve the following task:

Task: {task}

You **secretly** know the answer is: {answer}. 
And a normal reasoning process for inference is: {solution}
**However, you MUST pretend you don't know the answer and the reasoning process. And you need to generate the reasoning process and the answer by yourself.**
**For the coarse reasoning, you must be as concise as possible.**

You should return in the str format: 
Output format:\n**Coarse Reasoning**\n\n**Fine Reasoning**\n\n**Final Answer** \\boxed{{}}

Please return ONLY the str, nothing else! '''
    return prompt

def main():
    tain_data_pairs = list(load_jsonl("data/math/train.jsonl"))
    cot_queries = []
    for pair in tain_data_pairs:
        cot_queries.append(make_query_coarse2fine_cot_prompt(pair["problem"], pair["solution"], pair["answer"]))
        # break
    # import utils
    # answer = utils.retry_query_gptv2(cot_queries[0], "gpt-4o-mini")
    # print(answer)
    # import pdb;pdb.set_trace()
    query = MultiProcessingQuery("data/math-processed/math_train_coarse2fine.jsonl", model="gpt-4o-mini")
    query.query_a_list(cot_queries, worker_num=32)

if __name__ == '__main__':
    main()
