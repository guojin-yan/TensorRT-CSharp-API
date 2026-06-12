using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the name of the loop that owns this loop-boundary layer.
    /// 获取拥有此 loop-boundary 层的 loop 名称。
    /// </summary>
    public string GetLoopBoundaryLoopName()
    {
        return NativeBridgeApi.GetLoopBoundaryLoopName(Line, _handle);
    }

    /// <summary>
    /// Gets the name of the if-conditional that owns this conditional-boundary layer.
    /// 获取拥有此 conditional-boundary 层的 if-conditional 名称。
    /// </summary>
    public string GetIfConditionalBoundaryName()
    {
        return NativeBridgeApi.GetIfConditionalBoundaryName(Line, _handle);
    }

    /// <summary>
    /// Gets the loop-output layer output kind.
    /// 获取 loop-output 层的输出类型。
    /// </summary>
    public TensorRtLoopOutputKind GetLoopOutputKind()
    {
        return NativeBridgeApi.GetLoopOutputKind(Line, _handle);
    }

    /// <summary>
    /// Gets the loop-output concatenation axis.
    /// 获取 loop-output 的拼接轴。
    /// </summary>
    public int GetLoopOutputAxis()
    {
        return NativeBridgeApi.GetLoopOutputAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the loop-output concatenation axis.
    /// 设置 loop-output 的拼接轴。
    /// </summary>
    /// <param name="axis">The concatenation axis. / 拼接轴。</param>
    public void SetLoopOutputAxis(int axis)
    {
        NativeBridgeApi.SetLoopOutputAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the trip-limit kind for a trip-limit layer.
    /// 获取 trip-limit 层的限制方式。
    /// </summary>
    public TensorRtTripLimitKind GetTripLimitKind()
    {
        return NativeBridgeApi.GetTripLimitKind(Line, _handle);
    }

    /// <summary>
    /// Gets the iterator axis for an iterator layer.
    /// 获取 iterator 层的迭代轴。
    /// </summary>
    public int GetIteratorAxis()
    {
        return NativeBridgeApi.GetIteratorAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the iterator axis for an iterator layer.
    /// 设置 iterator 层的迭代轴。
    /// </summary>
    /// <param name="axis">The iterator axis. / 迭代轴。</param>
    public void SetIteratorAxis(int axis)
    {
        NativeBridgeApi.SetIteratorAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets whether an iterator layer iterates in reverse order.
    /// 获取 iterator 层是否按反向顺序迭代。
    /// </summary>
    public bool GetIteratorReverse()
    {
        return NativeBridgeApi.GetIteratorReverse(Line, _handle);
    }

    /// <summary>
    /// Sets whether an iterator layer iterates in reverse order.
    /// 设置 iterator 层是否按反向顺序迭代。
    /// </summary>
    /// <param name="reverse">Whether iteration should be reversed. / 是否反向迭代。</param>
    public void SetIteratorReverse(bool reverse)
    {
        NativeBridgeApi.SetIteratorReverse(Line, _handle, reverse);
    }

    /// <summary>
    /// Sets the int64 alpha value of a TensorRT fill layer.
    /// 设置 TensorRT fill 层的 int64 alpha 值。
    /// </summary>
    public void SetFillAlphaInt64(long value)
    {
        NativeBridgeApi.SetFillAlphaInt64(Line, _handle, value);
    }

    /// <summary>
    /// Gets the int64 alpha value of a TensorRT fill layer.
    /// 获取 TensorRT fill 层的 int64 alpha 值。
    /// </summary>
    public long GetFillAlphaInt64()
    {
        return NativeBridgeApi.GetFillAlphaInt64(Line, _handle);
    }

    /// <summary>
    /// Sets the int64 beta value of a TensorRT fill layer.
    /// 设置 TensorRT fill 层的 int64 beta 值。
    /// </summary>
    public void SetFillBetaInt64(long value)
    {
        NativeBridgeApi.SetFillBetaInt64(Line, _handle, value);
    }

    /// <summary>
    /// Gets the int64 beta value of a TensorRT fill layer.
    /// 获取 TensorRT fill 层的 int64 beta 值。
    /// </summary>
    public long GetFillBetaInt64()
    {
        return NativeBridgeApi.GetFillBetaInt64(Line, _handle);
    }

    /// <summary>
    /// Gets whether a TensorRT fill layer currently stores alpha and beta as int64 values.
    /// 获取 TensorRT fill 层当前是否以 int64 形式保存 alpha 和 beta。
    /// </summary>
    public bool IsFillAlphaBetaInt64()
    {
        return NativeBridgeApi.IsFillAlphaBetaInt64(Line, _handle);
    }
}
