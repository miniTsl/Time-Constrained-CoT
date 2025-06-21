from matplotlib import pyplot as plt
import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--prompt_type", type=str, default="qwen25-math-cot")
    # parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B-Preview")
    parser.add_argument("--data_name", type=str, default="gsm8k")
    # TODO: add more args
    return parser.parse_args()

def main():
    args = parse_args()
    # models=["Qwen/QwQ-32B-Preview", "Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
    models=["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
    prompt_types=["qwen25-math-cot", "coarse-to-fine-structured"]
    
    ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta", "teal", "lavender", "maroon", "mint", "coral", "gold", "silver", "bronze", "copper"]
    # plot acc vs ratio, mark the data point with red color
    # write the data value on the data point
    plt.figure(figsize=(12, 8))
    plt.yticks(range(0, 101, 5))  # 设置纵坐标的间隔为5
    
    for index, model in enumerate(models):
        dir_prefix1 = "../outputs/12_11/" + model + "/" + prompt_types[0] + "/" + args.data_name
        dir_prefix2 = "../outputs/12_11/" + model + "/" + prompt_types[1] + "/" + args.data_name
        file_prefix1 = "test_" + prompt_types[0] + "_-1_seed0_t0.0_s0_e-1"
        file_postfix1 = "_" + prompt_types[0] + "_metrics.json"
        file_prefix2 = "test_" + prompt_types[1] + "_-1_seed0_t0.0_s0_e-1"
        file_postfix2 = "_" + prompt_types[1] + "_metrics.json"
        gt_file_path1 = os.path.join(dir_prefix1, file_prefix1 + file_postfix1)
        gt_file_path2 = os.path.join(dir_prefix2, file_prefix2 + file_postfix2)
        
        acc_list1 = []
        acc_list2 = []
        for r in ratio_list:
            file_name1 = file_prefix1 + "_r" + str(r) + file_postfix1
            file_name2 = file_prefix2 + "_r" + str(r) + file_postfix2
            with open(os.path.join(dir_prefix1, file_name1), "r") as f:
                data = json.load(f)
                acc_list1.append(data["acc"])
            with open(os.path.join(dir_prefix2, file_name2), "r") as f:
                data = json.load(f)
            acc_list2.append(data["acc"])

            with open(gt_file_path1, "r") as f:
                gt_data = json.load(f)
                gt_acc1 = gt_data["acc"]
            with open(gt_file_path2, "r") as f:
                gt_data = json.load(f)
                gt_acc2 = gt_data["acc"]
    
        plt.plot(ratio_list, acc_list1, marker='o', color=colors[index], label=model.split('/')[-1] + "_step_by_step", linestyle='--')
        # for i, acc in enumerate(acc_list1):
        #     plt.text(ratio_list[i], acc, str(acc), ha='center', va='bottom', color=colors[index])
        
        plt.plot(ratio_list, acc_list2, marker='o', color=colors[index], label=model.split('/')[-1] + "_coarse-to-fine")
        # for i, acc in enumerate(acc_list2):
        #     plt.text(ratio_list[i], acc, str(acc), ha='center', va='bottom', color=colors[index])

        # # 添加真实准确率的绘制
        # plt.axhline(y=gt_acc1, color='blue', linestyle='--', label='Ground Truth Accuracy 1')
        # plt.axhline(y=gt_acc2, color='orange', linestyle='--', label='Ground Truth Accuracy 2')
        # plt.text(0.2, gt_acc1, str(gt_acc1), ha='right', va='bottom', color='blue')
        # plt.text(0.4, gt_acc2, str(gt_acc2), ha='right', va='bottom', color='orange')
    
    plt.legend()
    
    plt.xlabel("Ratio")
    plt.ylabel("Accuracy")
    plt.title(args.data_name + " Accuracy vs Ratio")
    plt.savefig(args.data_name + "_" + prompt_types[1] + ".png")

if __name__ == "__main__":
    main()