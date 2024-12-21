from sentence_transformers import SentenceTransformer
from utils import load_jsonl, save_pth, dump_json, load_json
import torch

def embed_train_data():
    data = list(load_jsonl("data/math/train.jsonl"))
    model = SentenceTransformer('all-mpnet-base-v2')
    problem_embeddings = []
    for i, item in enumerate(data):
        print(f"Processing {i}...")
        problem = item['problem']
        problem_embeddings.append(model.encode(problem))
    save_pth("data/math-processed/math_train_problem_embeddings.pth", problem_embeddings)

def get_topk_nearest_train_questions(problem, problem_embeddings, model, k=50):
    problem_embed = model.encode(problem)
    similarity = model.similarity(problem_embed, problem_embeddings)
    topk_indices = torch.topk(similarity, k).indices
    # convert to list
    topk_indices = topk_indices.tolist()
    return topk_indices

def main():
    # embed_train_data()
    model = SentenceTransformer('all-mpnet-base-v2')
    test_data = list(load_jsonl("data/math/test.jsonl"))
    train_data = list(load_jsonl("data/math/train.jsonl"))

    problem_embeddings = torch.load("data/math-processed/math_train_problem_embeddings.pth")

    all_test_data_neighbours = {}

    for i, item in enumerate(test_data):
        print(f"Processing {i}...")
        problem = item['problem']
        indices = get_topk_nearest_train_questions(problem, problem_embeddings, model)
        all_test_data_neighbours[problem] = indices
        # print(problem)
        # print('-------')
        # print(train_data[indices[0][0]]['problem'])
        # print('-'*30)
        # if i == 30:
        #     break
    
        dump_json("data/math-processed/math_test_neighbours.json", all_test_data_neighbours)
        
def post_process():
    neighbors = load_json("data/math-processed/math_test_neighbours.json")
    new_neighbors = {}
    for k, v in neighbors.items():
        new_neighbors[k] = v[0]
    dump_json("data/math-processed/math_test_neighbours.json", new_neighbors)


if __name__ == "__main__":
    # main()
    post_process()