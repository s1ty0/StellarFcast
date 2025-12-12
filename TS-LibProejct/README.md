# 虚拟环境的配置

按照`requirements.txt`配置即可(推荐用conda的指令新建虚拟环境，方便同`LLMProject`的虚拟环境作区分)

```
pip install -r requirements.txt
```





# 复现

对应的复现指令（注：脚本后面携带的数字，代表指定的`cuda gpu`序号）：

```
# 基础运行
sh ./scripts/classification/DLinear.sh 1 
sh ./scripts/classification/PatchTST.sh 4
sh ./scripts/classification/iTransformer.sh 2
sh ./scripts/classification/TimesNet.sh 5
sh ./scripts/classification/Informer.sh 3
sh ./scripts/classification/Autoformer.sh 4
sh ./scripts/classification/iTransformer.sh 4
sh ./scripts/classification/MICN.sh 4

# 改进开启后运行：
sh ./scripts/classification/DLinear.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/PatchTST.sh 4 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/iTransformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/TimesNet.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/Informer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/Autoformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/iTransformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/MICN.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
```



