# 虚拟环境的配置

按照`requirements.txt`配置即可(推荐用conda的指令新建虚拟环境，方便同`TS-LibProject`的虚拟环境作区分)

```
pip install -r requirements.txt
```

我不得不提醒你的是，如果在安装依赖的时候碰到了一些意外的错误，我推荐你询问大模型寻求帮助，实验所需的所有包已经列出，应该不会出现意外的错误。



# 预训练大语言模型的下载

我们实验中用了以下预训练大语言模型，在对应的`huggingface`下载到本地即可(对应的网址已经罗列在右面)，因为实验中采用的是从本地加载大语言模型的方式。

```
bert_base_uncased # https://huggingface.co/google-bert/bert-base-uncased
gpt2 # https://huggingface.co/openai-community/gpt2
MOMENT-1-large # https://huggingface.co/AutonLab/MOMENT-1-large
roberta-base # https://huggingface.co/FacebookAI/roberta-base
```

下载好后，确保当下目录存在以下文件夹及文件：

```
├── models
│   ├── bert_base_uncased
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── gpt2
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── MOMENT-1-large
│   └── roberta-base
│       ├── model.safetensors
│       ├── pytorch_model.bin
│       ├── rust_model.ot
│       ├── tf_model.h5
│       ├── tokenizer.json
│       └── vocab.json
├── myDataK # 提前构建好的数据集存放目录
├── myDataT # 提前构建好的数据集存放目录
```



# 复现

对应的复现指令如下：

```
# # 以下是改进点开启后的复现指令：
# Kepler数据上：
# gpt2 模型
CUDA_VISIBLE_DEVICES=4 python main.py --model_type gpt2 --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset kepler

# bert模型
CUDA_VISIBLE_DEVICES=4 python main.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset kepler

# roberta模型
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset kepler

# 尝试复现的FLARE
CUDA_VISIBLE_DEVICES=O python main_flare.py --dataset kepler



# TESS 数据上：
# gpt2 模型
CUDA_VISIBLE_DEVICES=4 python main.py --model_type gpt2 --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset tess

# bert模型
CUDA_VISIBLE_DEVICES=4 python main.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset tess

# roberta模型
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset tess

# 尝试复现的FLARE
CUDA_VISIBLE_DEVICES=O python main_flare.py --dataset tess
```



消融实验，用来判定各个模块对实验的影响

```
# kepler数据
# 关闭--on_mm_history
CUDA_VISIBLE_DEVICES=4 python main.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics  --dataset kepler

# --on_mm_statistics
CUDA_VISIBLE_DEVICES=4 python main.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance  --on_mm_history --dataset kepler

# 关闭--on_enhance
CUDA_VISIBLE_DEVICES=4 python main.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset kepler

# 关闭--on_phy_loss
CUDA_VISIBLE_DEVICES=4 python main.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --dataset kepler
```

