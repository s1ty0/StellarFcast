# 1. 配置环境

同样的，需要安装以下依赖：

```
pandas
scikit-learn
pyarrow
numpy
torch
transformers
tqdm
```

但是，我不推荐你再次新建一个虚拟环境，在绝大数基础的python环境（base）里，应该都满足上述条件。

若不满足，安装其中缺失的包即可。也可以构建一个虚拟环境，用来做数据处理。



# 2. 获取数据

首先，我们需要下载原始数据到`当下目录data文件夹中`，这包括两方面的数据：Kepler和TESS，数据我已经打包好，放在

1. Google: 「TODO」
2. 百度网盘: 「TODO」里面

其中Kepler数据引用自 https://huggingface.co/datasets/Maxwell-Jia/kepler_flare， TESS数据来源于我们的数据团队（现也已经公开）

注：如果直接在此处下载Kepler数据，你将得到原始数据，但我们对原始数据进行了简单的合并处理（代码可参考TODO），用于执行我们后续的数据处理的流程，当然，google和百度网盘里面存放的已经是合并好的数据，直接使用即可。



获得好数据后，请确保你的目录包含`data文件夹（及其子文件夹）`，然后，才可以执行脚本。

```
├── DataProcessProject
│   ├── data
│   │   ├── all.parquet
│   │   ├── my_data.pt
...
```



# 3. 执行数据处理脚本

然后，执行对应的数据脚本，进行patch 和clean的操作（同时避免了数据泄露）
在当面目录下依次执行：

```
sh ./data_pipeline_kepler.sh
sh ./data_pipeline_tess.sh
```

将会得到两个文件夹：

```
./myDataK # 存放得到的Kepler数据
./myDataT # 存放得到的TESS数据
```
这样，我们就完成了基础的数据准备工作。

等等！先别急，我们进行实验所必需的，一些改进点的引入，仍需要构建history和statistics的嵌入

首先下载实验所使用的文本编码器：`bert`,

```
python encoderDown.py
```

然后，执行如下脚本，得到history和statistics的嵌入：

```
sh ./data_pipeline_kepler.sh
sh ./data_pipeline_tess.sh
```

随后将这俩文件夹分别复制、粘贴到`LLMProject文件夹`和`TS-LibProject文件夹`下。就宣告数据处理章节的结束。
