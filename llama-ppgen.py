"""
LLaMa-2 PPGen 脚本
此脚本编译并运行 ppgen 工具以生成 LLaMa-2 模型的预处理参数文件。
参数:
    model_size (int): 要使用的模型大小。可选值为 7 或 13。默认值为 13。
    --log_off_factor (int): 要使用的日志偏移因子。默认值为 5。
用法:
    python llama-ppgen.py <model_size> [--log_off_factor <log_off_factor>]
示例:
    python llama-ppgen.py 13 --log_off_factor 5
依赖:
    - transformers (AutoTokenizer, AutoModelForCausalLM)
    - make (用于编译 ppgen)
步骤:
    1. 使用 'make ppgen' 命令编译 ppgen 工具。
    2. 从本地存储加载指定的 LLaMa-2 模型和分词器。
    3. 为指定的模型大小创建工作目录。
    4. 遍历模型的参数并使用 ppgen 生成预处理参数文件。
异常:
    ValueError: 如果参数具有意外的形状。
"""
import os, sys,argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description='LLaMa-2 PPGen')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('--log_off_factor', type=int, default=5, help='The log offset factor to use. Default is 5')

if __name__ == '__main__':
    compilation_error = os.system('make ppgen')
    if compilation_error:
        print("Error compiling ppgen")
        exit(1)
    args = parser.parse_args()
    model_card = f"/home/wenmou/code/zkllm-ccs2024/model-storage/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")

    os.makedirs(f"./zkllm-workdir/Llama-2-{args.model_size}b", exist_ok = True)

    for (i, w) in model.model.layers[0].named_parameters():
        if len(w.shape) == 2:
            pp_size = w.shape[0]
            pp_size <<= args.log_off_factor
        elif len(w.shape) == 1:
            (pp_size,) = w.shape
        else:
            raise ValueError(f"Unexpected shape {w.shape} for parameter {i}")
        
        os.system(f'./ppgen {pp_size} ./zkllm-workdir/Llama-2-{args.model_size}b/{i}-pp.bin')