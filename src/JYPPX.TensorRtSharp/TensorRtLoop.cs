using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a TensorRT loop definition owned by a network.
/// 表示由 network 持有生命周期的 TensorRT loop 定义。
/// </summary>
public sealed class TensorRtLoop : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtLoop(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this loop.
    /// 获取此 loop 使用的 TensorRT API 系列。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the TensorRT loop name.
    /// 获取或设置 TensorRT loop 名称。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetLoopName(Line, _handle);
        set => NativeBridgeApi.SetLoopName(Line, _handle, value);
    }

    /// <summary>
    /// Adds a recurrence boundary layer to the loop.
    /// 向 loop 添加 recurrence 边界层。
    /// </summary>
    /// <param name="initialValue">Initial tensor value for the recurrence. / recurrence 的初始张量值。</param>
    /// <returns>The network-owned recurrence layer. / 由 network 持有生命周期的 recurrence 层。</returns>
    public TensorRtLayer AddRecurrence(TensorRtTensor initialValue)
    {
        ValidateTensor(initialValue, nameof(initialValue));
        return new TensorRtLayer(Line, NativeBridgeApi.AddLoopRecurrence(Line, _handle, initialValue.Handle));
    }

    /// <summary>
    /// Adds a trip-limit boundary layer to the loop.
    /// 向 loop 添加 trip-limit 边界层。
    /// </summary>
    /// <param name="tensor">Scalar count or while-condition tensor. / 标量计数或 while 条件张量。</param>
    /// <param name="kind">Trip-limit interpretation. / trip-limit 解释方式。</param>
    /// <returns>The network-owned trip-limit layer. / 由 network 持有生命周期的 trip-limit 层。</returns>
    public TensorRtLayer AddTripLimit(TensorRtTensor tensor, TensorRtTripLimitKind kind)
    {
        ValidateTensor(tensor, nameof(tensor));
        return new TensorRtLayer(Line, NativeBridgeApi.AddLoopTripLimit(Line, _handle, tensor.Handle, kind));
    }

    /// <summary>
    /// Adds an iterator boundary layer to the loop.
    /// 向 loop 添加 iterator 边界层。
    /// </summary>
    /// <param name="tensor">Tensor to iterate over. / 要进行迭代的张量。</param>
    /// <param name="axis">Axis along which TensorRT iterates. / TensorRT 迭代的轴。</param>
    /// <param name="reverse">Whether to iterate in reverse order. / 是否反向迭代。</param>
    /// <returns>The network-owned iterator layer. / 由 network 持有生命周期的 iterator 层。</returns>
    public TensorRtLayer AddIterator(TensorRtTensor tensor, int axis = 0, bool reverse = false)
    {
        ValidateTensor(tensor, nameof(tensor));
        return new TensorRtLayer(Line, NativeBridgeApi.AddLoopIterator(Line, _handle, tensor.Handle, axis, reverse));
    }

    /// <summary>
    /// Adds a loop-output boundary layer.
    /// 添加 loop-output 边界层。
    /// </summary>
    /// <param name="tensor">Tensor produced inside the loop. / loop 内部产生的张量。</param>
    /// <param name="kind">Loop output semantics. / loop 输出语义。</param>
    /// <param name="axis">Concatenation axis for concatenate-style outputs. / 拼接类输出使用的轴。</param>
    /// <returns>The network-owned loop-output layer. / 由 network 持有生命周期的 loop-output 层。</returns>
    public TensorRtLayer AddOutput(TensorRtTensor tensor, TensorRtLoopOutputKind kind, int axis = 0)
    {
        ValidateTensor(tensor, nameof(tensor));
        return new TensorRtLayer(Line, NativeBridgeApi.AddLoopOutput(Line, _handle, tensor.Handle, kind, axis));
    }

    /// <summary>
    /// Releases the managed bridge reference for this network-owned loop.
    /// 释放此 network-owned loop 的托管桥接引用。
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
            throw new ArgumentException("Tensor must belong to the same TensorRT API line as the loop.", parameterName);
        }
    }
}
