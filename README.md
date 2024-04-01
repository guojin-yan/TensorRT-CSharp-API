![TensorRTSharp](https://socialify.git.ci/guojin-yan/TensorRT-CSharp-API/image?description=1&descriptionEditable=TensorRT%20wrapper%20for%20.NET&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F04%2F11%2FOtsq6zAaZnwxP1U.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

##  📚简介

&emsp;    NVIDIA® TensorRT™ 是一款用于高性能深度学习推理的 SDK，包括深度学习推理优化器和运行时，可为推理应用程序提供低延迟和高吞吐量。基于 NVIDIA TensorRT 的应用程序在推理过程中的执行速度比纯 CPU 平台快 36 倍，使您能够优化在所有主要框架上训练的神经网络模型，以高精度校准低精度，并部署到超大规模数据中心、嵌入式平台或汽车产品平台。

<div align=center><img src="https://s2.loli.net/2024/04/01/oAB1m9XWR4gpJaG.png" width=400></div>

&emsp;    TensorRT 基于 NVIDIA CUDA® 并行编程模型构建，使您能够在 NVIDIA GPU 上使用量化、层和张量融合、内核调整等技术来优化推理。TensorRT 提供 INT8 使用量化感知训练和训练后量化和浮点 16 （FP16） 优化，用于部署深度学习推理应用程序，例如视频流、推荐、欺诈检测和自然语言处理。低精度推理可显著降低延迟，这是许多实时服务以及自主和嵌入式应用所必需的。TensorRT 与 PyTorch 和 TensorFlow 集成，因此只需一行代码即可实现 6 倍的推理速度。TensorRT 提供了一个 ONNX 解析器，因此您可以轻松地将 ONNX 模型从常用框架导入 TensorRT。它还与 ONNX 运行时集成，提供了一种以 ONNX 格式实现高性能推理的简单方法。

&emsp;    基于这些优势，TensorRT目前在深度模型部署应用越来越广泛。但是TensorRT目前只提供了C++与Python接口，对于跨语言使用十分不便。目前C#语言已经成为当前编程语言排行榜上前五的语言，也被广泛应用工业软件开发中。为了能够实现在C#中调用TensorRT部署深度学习模型，我们在之前的开发中开发了TensorRT C# API。虽然实现了该接口，但由于数据传输存在问题，当时开发的版本在应用时存在较大的问题。

&emsp;    基于此，我们开发了TensorRT C# API 2.0版本，该版本在开发时充分考虑了上一版本应用时出现的问题，并进行了改进。同时在本版本中，我们对接口进行了优化，使用起来更加简单，并同时提供了相关的应用案例，方便开发者进行使用。

##  ⚙安装

### TensorRT安装

TensorRT依赖于CUDA加速，因此需要同时安装CUDA与TensorRT才可以使用，且CUDA与TensorRT版本之间需要对应，否者使用会出现较多问题，因此此处并未提供Nuget包，组要根据自己电脑配置选择合适的版本安装后重新编译本项目源码，

## 💻 应用案例

获取耕读应用案例请参考：[TensorRT-CSharp-API-Samples](https://github.com/guojin-yan/TensorRT-CSharp-API-Samples.git)

## 🗂 API文档

### 命名空间

```c#
using TensorRTSharp;
using TensorRtSharp.Custom;
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
	    <td rowspan="3" align="center">1</td>
	    <td align="center">方法</td>
        <td>OnnxToEngine()</td>
        <td>将onnx模型转为engine</td>
        <td rowspan="3">可以调用封装的TensorRT中的ONNX 解释器，对ONNX模型进行转换，并根据本机设备信息，编译本地模型，将模型转换为TensorRT 支持的engine格式。</td>
	</tr>
    <tr >
	    <td rowspan="2" align="center">参数</td>
        <td><font color=blue>string</font> modelPath</td>
        <td>本地ONNX模型地址，只支持ONNX格式，且ONNX模型必须为确定的输入输出，暂不支持动态输入。</td>
	</tr>
    <tr >
        <td><font color=blue>int</font> memorySize</td>
        <td>模型转换时分配的内存大小</td>
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
        <td>Nvinfer()</td>
        <td>构造函数/初始化函数</td>
        <td rowspan="2">初始化Nvinfer类，主要初始化封装的推理引擎，该推理引擎中封装了比较重要的一些类和指针。</td>
	</tr>
    <tr >
	    <td rowspan="1" align="center">参数</td>
        <td><font color=blue>string</font> modelPath</td>
        <td>- -  engine模型路径。</td>
	</tr>
	<tr >
	    <td rowspan="4" align="center">2</td>
	    <td align="center">方法</td>
        <td><font color=blue>Dims </font> GetBindingDimensions()</td>
        <td>获取绑定的端口的形状信息</td>
        <td rowspan="4">通过端口编号或者端口名称，获取绑定的端口的形状信息.</td>
	</tr>
    <tr >
	    <td rowspan="2" align="center">参数</td>
        <td><font color=blue>int</font> index</td>
        <td>绑定端口的编号</td>
	</tr>
    <tr >
        <td><font color=blue>string</font> nodeName</td>
        <td>- 绑定端口的名称</td>
	</tr>
    <tr >
	    <td rowspan="1" align="center">返回值</td>
        <td><font color=blue>Dims</td>
        <td>接口返回一个**Dims**结构体，该结构体包含了节点的维度大小以及每个维度的具体大小。
        </td>
	</tr>
	<tr >
	    <td rowspan="4" align="center">3</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> LoadInferenceData()</td>
        <td>加载待推理数据接口</td>
        <td rowspan="4">通过端口编号或者端口名称，将处理好的带推理数据加载到推理通道上。</td>
	</tr>
  <tr >
	    <td rowspan="3" align="center">参数</td>
        <td><font color=blue>string</font> nodeName</td>
        <td>待加载推理数据端口的名称。</td>
	</tr>
    <tr >
        <td><font color=blue>int</font> nodeIndex</td>
        <td>待加载推理数据端口的编号。</td>
	</tr>
    <tr >
        <td><font color=blue>float[]</font> data</td>
        <td>处理好的待推理数据，由于目前使用的推理数据多为float类型，因此此处目前只做了该类型接口。</td>
	</tr>
    <tr >
	    <td rowspan="1" align="center">4</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> infer()</td>
        <td> 调用推理接口，对加载到推理通道的数据进行推理。</td>
	</tr>
    <tr >
	    <td rowspan="4" align="center">5</td>
	    <td align="center">方法</td>
        <td><font color=blue>void</font> LoadInferenceData()</td>
        <td>获取推理结果:</td>
        <td rowspan="4">通过端口编号或者端口名称，读取推理好的结果数据。</td>
	</tr>
  <tr >
	    <td rowspan="2" align="center">参数</td>
        <td><font color=blue>string</font> nodeName</td>
        <td>推理结果数据端口的名称。</td>
	</tr>
    <tr >
        <td><font color=blue>int</font> nodeIndex</td>
        <td>推理结果数据端口的编号。</td>
	</tr>
    <tr >
	    <td rowspan="1" align="center">返回值</td>
        <td><font color=blue> float[]</td>
        <td>返回值为指定节点的推理结果数据。</td>
	</tr>

