from sentence_transformers import SentenceTransformer
from utils import load_jsonl, save_pth
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
    return topk_indices

def main():
    # embed_train_data()
    model = SentenceTransformer('all-mpnet-base-v2')
    test_data = list(load_jsonl("data/math/test.jsonl"))

    problem_embeddings = torch.load("data/math-processed/math_train_problem_embeddings.pth")

    for i, item in enumerate(test_data):
        problem = item['problem']
        indices = get_topk_nearest_train_questions(problem, problem_embeddings, model)



if __name__ == "__main__":
    main()
        