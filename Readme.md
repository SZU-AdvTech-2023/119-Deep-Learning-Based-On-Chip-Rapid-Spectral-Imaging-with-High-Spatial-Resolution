# ADMM-net

v1.0.0

### 

### 1.配置文件

```json
{
  "name": "condata",
  "version": "1.0.0",
  "train": {
    "batch_size": 8,
    "learning_rate": 0.001,
    "epoch": 300,
    "in_channel": 26,
    "out_channel": 26,
    "degree": 9,
    "img_width": 64
	"SNR": 90,
    "alpha": 0
  }
}
```

| 变量名        | 含义                                             |
| ------------- | ------------------------------------------------ |
| batch_size    | 批次大小，在cia003上可以64*64大小跑8批次         |
| learning_rate | 学习率                                           |
| epoch         | 轮数                                             |
| in_channel    | 输入通道数，λ                                    |
| out_channel   | 输出通道数                                       |
| degree        | \Phi值曲线条数                                   |
| img_width     | 输入像素大小，(img_width, img_width, in_channel) |
| SNR           | 信噪比                                           |
| alpha         | 信号相关和无关噪声在总噪声中的占比               |

### 2.训练命令

train.py：训练代码

```python
python --data_path PATH
```

### 3.训练输出

输出结果在./run/{time}/下，其中包含了config、logs、Phi、models等文件夹，分别保存了不同的运行输出。

### 4.其他代码及功能

| 文件名            | 路径                   | 作用                                                         |
| ----------------- | ---------------------- | ------------------------------------------------------------ |
| ADMM_net          | ./ADMM_net.py          | 网络主体部分，包含了线性变换和去噪网络                       |
| cave_pngtoimg     | ./cave_pngtoimg.py     | 将cave数据集各通道png保存为ENVI形式                          |
| change200         | ./change200.py         | 将kaist数据集（此处数据集已经以ENVI形式保存）随机选择200个（512,512,26）大小的文件 |
| dataset           | ./dataset.py           | 将ENVI格式的文件读取并封装为dataset                          |
| desam128_to64     | ./desam128_to64.py     | 降采样                                                       |
| kaist_desamandcut | ./kaist_desamandcut.py | 对kaist数据集降采样并采集                                    |
| kaist_exrtoimg    | ./kaist_exrtoimg.py    | 将原始KAISTexr格式转为ENVI格式                               |
| load              | ./load.py              | 测试模型（暂时只写了保真率)                                  |
| train             | ./train.py             | 训练主体代码                                                 |
| img_resolve       | ./utils/img_resolve.py | 读取和保存ENVI格式的数据                                     |

