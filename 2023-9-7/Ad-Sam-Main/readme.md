# 1. 安装环境
```
conda create -n control python=3.8  
conda activate control  
pip install -r requirement
```
# 2. 准备数据
下载[sam-1b](https://opendatalab.com/OpenDataLab/SA-1B)数据集, 由于**sam-1b**巨大, 我们只下载三个子集，如图所示:
<img width=300 src=/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/property/4.png>


# 3. 下载Stable Diffusion预训练权重(这里版本先选择1.5，后面会升级到XL) 
## 方式一: 手动下载，
可以选择HuggingFace(https://huggingface.co/runwayml/stable-diffusion-v1-5) 或 ModelScope(https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files), 将下载好后的权重放在ckpt文件夹下，如图所示:  
<img width=300 src=property/1.png>

## 方式二: 自动下载，
需要服务器能连接huggingface, 如果使用这种方式，请将 grad_null_text_inversion_edit.py 和 null_text_inversion.py里的stable diffusion加载方式修改下，具体做法就是将"ckpt改为"runwayml"(null_text_inversion.py里的667行，grad_null_text_inversion_edit.py里的760行),如图所示。

修改前:  
<img width=300 src=property/2.png>

修改后:  
<img width=300 src=property/3.png>


# 4. 运行
## 步骤1: 反演  
```
./inversion.sh
```

## 步骤2: 对抗样本生成  
```
./grad.sh
```




