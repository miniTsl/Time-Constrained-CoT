``0212/``文件夹下的csv文件是不同模型在input分别为150、200、250个token下，输出分别是64 tokens, 128 tokens, 256 tokens, 512 tokens, 1024 tokens的latency，构成tokens-latency之间的映射

``latency_mapping.py``是用来运行模型并得到这些映射关系的代码

