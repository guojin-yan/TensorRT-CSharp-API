using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Gets whether a native TensorRT error recorder is attached to this network definition.
    /// 获取当前 network definition 是否绑定了 TensorRT 原生 error recorder；支持 TensorRT 8/10/11，不会暴露 recorder 指针或接管其生命周期。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasNetworkErrorRecorder(Line, _handle);

    /// <summary>
    /// Clears the native TensorRT error recorder attached to this network definition.
    /// 清除当前 network definition 上绑定的 TensorRT 原生 error recorder；支持 TensorRT 8/10/11，不会销毁 recorder 或接管其生命周期。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearNetworkErrorRecorder(Line, _handle);
    }

    /// <summary>
    /// Removes a tensor from the TensorRT 11 network definition.
    /// 从 TensorRT 11 network definition 中移除一个 tensor。
    /// </summary>
    /// <param name="tensor">The tensor to remove. / 要移除的 tensor。</param>
    /// <remarks>
    /// TensorRT only permits removing tensors that are not used as layer inputs or outputs. The bridge forwards TensorRT's validation result.
    /// TensorRT 只允许移除没有被 layer 输入或输出使用的 tensor；桥接层会直接转发 TensorRT 的校验结果。
    /// </remarks>
    public void RemoveTensor(TensorRtTensor tensor)
    {
        ValidateInputTensor(tensor, nameof(tensor));
        NativeBridgeApi.RemoveNetworkTensor(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Adds the TensorRT 11 five-argument TopK layer variant with explicit output-indices type.
    /// 添加 TensorRT 11 五参数 TopK layer 变体，并显式指定输出 indices 类型。
    /// </summary>
    /// <param name="input">The input tensor. / 输入 tensor。</param>
    /// <param name="operation">The TopK operation. / TopK 操作类型。</param>
    /// <param name="k">The static K value. / 静态 K 值。</param>
    /// <param name="axes">The reduction axes bitmask. / reduce axes 位掩码。</param>
    /// <param name="indicesType">The output indices tensor type. Only <see cref="TensorRtDataType.Int32"/> and <see cref="TensorRtDataType.Int64"/> are valid. / 输出 indices tensor 类型，仅支持 <see cref="TensorRtDataType.Int32"/> 和 <see cref="TensorRtDataType.Int64"/>。</param>
    /// <returns>The created TopK layer. / 创建出的 TopK layer。</returns>
    public TensorRtLayer AddTopKV2(TensorRtTensor input, TensorRtTopKOperation operation, int k, uint axes, TensorRtDataType indicesType)
    {
        ValidateInputTensor(input, nameof(input));
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "TopK k must be greater than zero.");
        }

        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "TopK axes bitmask must not be zero.");
        }

        if (indicesType != TensorRtDataType.Int32 && indicesType != TensorRtDataType.Int64)
        {
            throw new ArgumentOutOfRangeException(nameof(indicesType), "TopK V2 indices type must be Int32 or Int64.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddTopKV2Layer(Line, _handle, input.Handle, operation, k, axes, indicesType));
    }
}
