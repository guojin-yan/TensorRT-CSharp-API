using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a TensorRT if-conditional definition owned by a network.
/// 表示由 network 持有生命周期的 TensorRT if-conditional 定义。
/// </summary>
public sealed class TensorRtIfConditional : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtIfConditional(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this conditional.
    /// 获取此 conditional 使用的 TensorRT API 系列。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the TensorRT conditional name.
    /// 获取或设置 TensorRT conditional 名称。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetIfConditionalName(Line, _handle);
        set => NativeBridgeApi.SetIfConditionalName(Line, _handle, value);
    }

    /// <summary>
    /// Sets the boolean condition tensor for this conditional.
    /// 设置此 conditional 的布尔条件张量。
    /// </summary>
    /// <param name="condition">Scalar boolean condition tensor. / 标量布尔条件张量。</param>
    /// <returns>The network-owned condition layer. / 由 network 持有生命周期的 condition 层。</returns>
    public TensorRtLayer SetCondition(TensorRtTensor condition)
    {
        ValidateTensor(condition, nameof(condition));
        return new TensorRtLayer(Line, NativeBridgeApi.SetIfConditionalCondition(Line, _handle, condition.Handle));
    }

    /// <summary>
    /// Adds a conditional input boundary layer.
    /// 添加 conditional input 边界层。
    /// </summary>
    /// <param name="input">Input tensor shared with conditional branches. / 与 conditional 分支共享的输入张量。</param>
    /// <returns>The network-owned conditional input layer. / 由 network 持有生命周期的 conditional input 层。</returns>
    public TensorRtLayer AddInput(TensorRtTensor input)
    {
        ValidateTensor(input, nameof(input));
        return new TensorRtLayer(Line, NativeBridgeApi.AddIfConditionalInput(Line, _handle, input.Handle));
    }

    /// <summary>
    /// Adds a conditional output boundary layer.
    /// 添加 conditional output 边界层。
    /// </summary>
    /// <param name="trueOutput">Output tensor from the true branch. / true 分支输出张量。</param>
    /// <param name="falseOutput">Output tensor from the false branch. / false 分支输出张量。</param>
    /// <returns>The network-owned conditional output layer. / 由 network 持有生命周期的 conditional output 层。</returns>
    public TensorRtLayer AddOutput(TensorRtTensor trueOutput, TensorRtTensor falseOutput)
    {
        ValidateTensor(trueOutput, nameof(trueOutput));
        ValidateTensor(falseOutput, nameof(falseOutput));
        return new TensorRtLayer(Line, NativeBridgeApi.AddIfConditionalOutput(Line, _handle, trueOutput.Handle, falseOutput.Handle));
    }

    /// <summary>
    /// Releases the managed bridge reference for this network-owned conditional.
    /// 释放此 network-owned conditional 的托管桥接引用。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateTensor(TensorRtTensor? tensor, string parameterName)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(parameterName);
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Tensor must belong to the same TensorRT API line as the conditional.", parameterName);
        }
    }
}
