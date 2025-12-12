# 虚拟环境的配置

按照`requirements.txt`配置即可(推荐用conda的指令新建虚拟环境，方便同`TS-LibProject`的虚拟环境作区分)

```
pip install -r requirements.txt
```



# 复现

对应的复现指令如下：

Kepler数据上：

```
# 以下是改进点开启后的复现指令：
# gpt2 模型
CUDA_VISIBLE_DEVICES=4 python myMain_his_kepler.py --model_type gpt2 --use_lora --exp_num 1 --encoder bert-chinese --on_phy_losson_enhance --on_mm_statistics --on_mm_history

# bert模型
CUDA_VISIBLE_DEVICES=4 python myMain_his_kepler.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_losson_enhance --on_mm_statistics --on_mm_history

# roberta模型
CUDA_VISIBLE_DEVICES=4 python myMain_his_kepler.py --model_type roberta --use_lora --exp_num 1 --encoder bert-chinese --on_phy_losson_enhance --on_mm_statistics --on_mm_history

# 尝试复现的FLARE
CUDA_VISIBLE_DEVICES=O python myMain_flare.py
```



TESS数据上（为了方便，我们仅仅修改了数据集的加载路径这一条，这也是 `myMain_his.py`和 `myMain_his_kepler.py`唯一区别的地方: `myMain_his.py`对应TESS数据，`myMain_his_kepler.py`对应Kepler数据）「」TODO，这里完全可以合并。

```
# 以下是改进点开启后的复现指令：
# gpt2 模型
CUDA_VISIBLE_DEVICES=4 python myMain_his_kepler.py --model_type gpt2 --use_lora --exp_num 1 --encoder bert-chinese --on_phy_losson_enhance --on_mm_statistics --on_mm_history

# bert模型
CUDA_VISIBLE_DEVICES=4 python myMain_his_kepler.py --model_type bert --use_lora --exp_num 1 --encoder bert-chinese --on_phy_losson_enhance --on_mm_statistics --on_mm_history

# roberta模型
CUDA_VISIBLE_DEVICES=4 python myMain_his_kepler.py --model_type roberta --use_lora --exp_num 1 --encoder bert-chinese --on_phy_losson_enhance --on_mm_statistics --on_mm_history

# 尝试复现的FLARE
CUDA_VISIBLE_DEVICES=O python myMain_flare.py
```

