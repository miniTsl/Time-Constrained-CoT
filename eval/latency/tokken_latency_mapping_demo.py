import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_paths =[
    #"0210/Llama-3.2-1B-Instruct_a800.csv",
    #"0210/Llama-3.2-3B-Instruct_a800.csv",
    #"0210/Qwen2.5-7B-Instruct_a800.csv",
    "0212/Qwen2.5-14B-Instruct_a800.csv",
    "0212/QwQ-32B-Preview_a800.csv",
    "0212/Llama-3.1-8B-Instruct_a800.csv"
]

Map_path_model={
    "0212/Qwen2.5-14B-Instruct_a800.csv":"Qwen-2.5-14B",
    "0212/QwQ-32B-Preview_a800.csv":"QwQ",
    "0212/Llama-3.1-8B-Instruct_a800.csv":"Llama-3.1-8B"
}

if __name__ == "__main__":

    # 创建一个图形
    plt.figure(figsize=(10, 8))

    colormaps = [plt.cm.Blues, plt.cm.Reds , plt.cm.Greens , plt.cm.Purples] 
    
    for idx,file_path in enumerate(file_paths):
        #print(idx)
        df = df = pd.read_csv(file_path, header=None)
        # 提取 x 轴标签
        x_labels = df.iloc[0, 1:].tolist()
        # 提取 x 轴标签中的数值
        x_values = [int(label.split()[0]) for label in x_labels]
        # 提取数据和行标签
        data = df.iloc[1:, 1:].values.tolist()
        #print(data)
        data = [[float(data[i][j]) for j in range(len(data[i]))] for i in range(len(data))]
        row_labels = df.iloc[1:, 0].tolist()
        # 绘制每一行数据对应的直线
        
        # 获取当前文件对应的颜色映射
        cmap = colormaps[idx % len(colormaps)]
        # 计算颜色的步长
        num_lines = len(data)
        color_step = 1.0 / (num_lines + 1)
        
        model = Map_path_model[file_path]
        print(model)
        for i, row in enumerate(data):
            #print(i,row)
            combined_label = model + " " + "input token" + ":" + row_labels[i].split(' ')[0]
            color = cmap((i + 1) * color_step)  # 获取颜色
            plt.plot(x_values, row, marker='o', label=combined_label,color= color)

    # 添加标题和标签
    #plt.title('The corresponding relationship between the input and output lengths and latency of LLMs.')
    plt.xlabel('output token', fontsize=25)
    plt.ylabel('Latency (s)', fontsize=25)

    # 设置 x 轴刻度和标签
    plt.xticks(x_values, x_values,fontsize=21)
    plt.yticks(fontsize=25)

    plt.legend(fontsize=17)
    
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1)

    # 保存图形为文件
    plt.savefig('inoutput_mapping_latency.pdf', dpi=300)