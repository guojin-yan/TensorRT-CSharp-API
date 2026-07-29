using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT 8 RNNv2 operation kinds.
/// 表示 TensorRT 8 RNNv2 运算类型。
/// </summary>
public enum TensorRtRnnOperation
{
    /// <summary>
    /// Single-gate RNN with ReLU activation.
    /// 使用 ReLU 激活的单门 RNN。
    /// </summary>
    Relu = 0,

    /// <summary>
    /// Single-gate RNN with tanh activation.
    /// 使用 tanh 激活的单门 RNN。
    /// </summary>
    Tanh = 1,

    /// <summary>
    /// Four-gate LSTM network without peephole connections.
    /// 不含 peephole 连接的四门 LSTM 网络。
    /// </summary>
    Lstm = 2,

    /// <summary>
    /// Three-gate gated recurrent unit network.
    /// 三门 GRU 网络。
    /// </summary>
    Gru = 3
}
/// <summary>
/// Represents TensorRT 8 RNNv2 direction modes.
/// 表示 TensorRT 8 RNNv2 方向模式。
/// </summary>
public enum TensorRtRnnDirection
{
    /// <summary>
    /// Iterate from the first input to the last input.
    /// 从第一个输入迭代到最后一个输入。
    /// </summary>
    Unidirection = 0,

    /// <summary>
    /// Iterate in both directions and concatenate outputs.
    /// 双向迭代并拼接输出。
    /// </summary>
    Bidirection = 1
}

/// <summary>
/// Represents TensorRT 8 RNNv2 input modes.
/// 表示 TensorRT 8 RNNv2 输入模式。
/// </summary>
public enum TensorRtRnnInputMode
{
    /// <summary>
    /// Perform the normal matrix multiplication in the first recurrent layer.
    /// 在第一个 recurrent layer 执行常规矩阵乘法。
    /// </summary>
    Linear = 0,

    /// <summary>
    /// Skip the first recurrent layer input matrix multiplication.
    /// 跳过第一个 recurrent layer 的输入矩阵乘法。
    /// </summary>
    Skip = 1
}

/// <summary>
/// Represents an individual TensorRT 8 RNNv2 gate.
/// 表示 TensorRT 8 RNNv2 单个门类型。
/// </summary>
public enum TensorRtRnnGateType
{
    /// <summary>Input gate (i). 输入门（i）。</summary>
    Input = 0,

    /// <summary>Output gate (o). 输出门（o）。</summary>
    Output = 1,

    /// <summary>Forget gate (f). 遗忘门（f）。</summary>
    Forget = 2,

    /// <summary>Update gate (z). 更新门（z）。</summary>
    Update = 3,

    /// <summary>Reset gate (r). 重置门（r）。</summary>
    Reset = 4,

    /// <summary>Cell gate (c). 单元门（c）。</summary>
    Cell = 5,

    /// <summary>Hidden gate (h). 隐状态门（h）。</summary>
    Hidden = 6
}
