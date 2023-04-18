![TensorRTSharp](https://socialify.git.ci/guojin-yan/TensorRTSharp/image?description=1&descriptionEditable=💞TensorRT%20wrapper%20for%20.NET💞&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F04%2F11%2FOtsq6zAaZnwxP1U.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

##  <img title="更新日志" src="https://s2.loli.net/2023/01/26/Zs1VFUT4BGQgfE9.png" alt="" width="40">简介

&emsp;   NVIDIA®TensorRT的核心™ 是一个C++库，有助于在NVIDIA图形处理单元（GPU）上进行高性能推理。TensorRT采用一个经过训练的网络，该网络由一个网络定义和一组经过训练的参数组成，并生成一个高度优化的运行时引擎，为该网络执行推理。TensorRT通过C++和Python提供API，帮助通过网络定义API表达深度学习模型，或通过解析器加载预定义模型，从而使TensorRT能够在NVIDIA GPU上优化和运行它们。TensorRT应用了图优化、层融合等优化，同时还利用高度优化的内核的不同集合找到了该模型的最快实现。TensorRT还提供了一个运行时，您可以使用该运行时在开普勒一代以后的所有NVIDIA GPU上执行该网络。TensorRT还包括Tegra中引入的可选高速混合精度功能™ X1，并用Pascal™, Volta™, Turing™, and NVIDIA® Ampere GPU 架构。

&emsp;   在推理过程中，基于 TensorRT 的应用程序的执行速度可比 CPU 平台的速度快 40 倍。借助 TensorRT，您可以优化在所有主要框架中训练的神经网络模型，精确校正低精度，并最终将模型部署到超大规模数据中心、嵌入式或汽车产品平台中。

&emsp;    官方发行的 TensorRT未提供C#编程语言接口，因此在使用时无法实现在C#中利用 TensorRT进行模型部署。在该项目中，利用动态链接库功能，调用官方依赖库，实现在C#中部署深度学习模型。

<img title="更新日志" src="https://s2.loli.net/2023/04/11/Otsq6zAaZnwxP1U.png" alt="" width="300">

## <img title="安装" src="https://s2.loli.net/2023/01/26/bm6WsE5cfoVvj7i.png" alt="" width="50"> 安装

### TensorRT安装

TensorRT依赖于CUDA加速，因此需要同时安装CUDA与TensorRT才可以使用，且CUDA与TensorRT版本之间需要对应，否者使用会出现较多问题，因此此处并未提供Nuget包，组要根据自己电脑配置悬着合适的版本安装后重新编译本项目源码，下面是TensorRT安装教程：[【TensorRT】NVIDIA TensorRT 安装 (Windows C++)_椒颜皮皮虾྅的博客-CSDN博客](https://blog.csdn.net/Grape_yan/article/details/127320959)



## <img title="API文档" src="https://s2.loli.net/2023/01/26/CNgHGrJ2DyvsaP4.png" alt="" width="50">API文档

### 命名空间

```c#
using TensorRTSharp;
```



### 模型推理API

<table>
	<tr>
	    <th width="7%" align="center" bgcolor=#FF7A68>序号</th>
	    <th width="35%" colspan="2" align="center" bgcolor=#FF7A68>API</th>
	    <th width="43%" align="center" bgcolor=#FF7A68>参数解释</th>  
        <th width="15%" align="center" bgcolor=#FF7A68>说明</th>
	</tr >
	<tr >
	    <td rowspan="4" align="center">1</td>
	    <td align="center">方法</td>
        <td>onnx_to_engine()</td>
        <td>将onnx模型转为engine</td>
        <td rowspan="4">将onnx模型转为engine格式，并按照设置转换模型精度</td>
	</tr>
    <tr >
	    <td rowspan="3" align="center">参数</td>
        <td><font color=blue>string</font> onnx_file_path</td>
        <td>ONNX模型路径</td>
	</tr>
    <tr >
        <td><font color=blue>string</font> engine_file_path</td>
        <td>输出模型路径</td>
	</tr>
    <tr >
        <td><font color=blue>string</font> type</td>
        <td>模型精度</td>
	</tr>

### 模型推理API

<table>
	<tr>
	    <th width="7%" align="center" bgcolor=#FF7A68>序号</th>
	    <th width="35%" colspan="2" align="center" bgcolor=#FF7A68>API</th>
	    <th width="43%" align="center" bgcolor=#FF7A68>参数解释</th>  
        <th width="15%" align="center" bgcolor=#FF7A68>说明</th>
	</tr >
	<tr >
	    <td rowspan="2" align="center">1</td>
	    <td align="center">方法</td>
        <td>Nvinfer()/init()</td>
        <td>构造函数/初始化函数</td>
        <td rowspan="2">初始化推理核心，读取本地engine模型</td>
	</tr>
    <tr >
	    <td rowspan="1" align="center">参数</td>
        <td><font color=blue>string</font> engine_filename</td>
        <td>模型路径</td>
	</tr>
	<tr >
	    <td rowspan="1" align="center">2</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> creat_gpu_buffer()</td>
        <td>创建gpu显存缓存</td>
	</tr>
	<tr >
	    <td rowspan="5" align="center">3</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> load_image_data()</td>
        <td>设置图片输入数据</td>
        <td rowspan="5">载入图片数据</td>
	</tr>
    <tr >
	    <td rowspan="4" align="center">参数</td>
        <td><font color=blue>string</font> node_name</td>
        <td>输入节点名称</td>
	</tr>
    <tr >
        <td><font color=blue>byte[]</font> image_data</td>
        <td>输入数据</td>
	</tr>
    <tr >
        <td><font color=blue>ulong</font> image_size</td>
        <td>图片大小</td>
	</tr>
    <tr >
        <td><font color=blue>BNFlag</font> BN_means</td>
        <td>数据处理类型：<br>type = 0: 均值方差归一化、常规缩放<br>type = 1: 普通归一化(1/255)、常规缩放<br>type = 2: 不归一化、常规缩放<br>type = 0: 均值方差归一化、仿射变换<br>type = 1: 普通归一化(1/255)、仿射变换<br>type = 2: 不归一化、仿射变换</td>
	</tr>
	<tr >
	    <td rowspan="1" align="center">4</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> infer()</td>
        <td>模型推理</td>
        <td rowspan="1"></td>
	</tr>
	<tr >
	    <td rowspan="3" align="center">5</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> read_infer_result()</td>
        <td>读取推理结果数据</td>
        <td rowspan="3">支持读取Float32</td>
	</tr>
    <tr >
	    <td rowspan="2" align="center">参数</td>
        <td><font color=blue>string</font> node_name</td>
        <td>输出节点名</td>
	</tr>
    <tr >
        <td><font color=blue>int</font> data_length</td>
        <td>输出数据长度</td>
	</tr>
	<tr >
	    <td rowspan="1" align="center">6</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> delete()</td>
        <td>删除内存地址</td>
        <td rowspan="1"></td>
	</tr>
