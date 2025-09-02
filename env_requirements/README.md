# Requirements

Before the evaluation, please install the required packages with the following command:

```bash
cd env_requirements
conda env create -f environment.yml

cd ../latex2sympy
pip install -e .
cd ../env_requirements
pip install -r pip_requirements.txt 
pip install flash-attn==2.7.0.post2
```
