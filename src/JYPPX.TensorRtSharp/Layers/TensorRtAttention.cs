using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a TensorRT 11 attention object owned by a network.
/// 表示由 network 持有生命周期的 TensorRT 11 attention 对象。
/// </summary>
public sealed class TensorRtAttention : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtAttention(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this attention.
    /// 获取此 attention 使用的 TensorRT API 系列。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the TensorRT diagnostic name for the attention.
    /// 获取或设置此 attention 的 TensorRT 诊断名称。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetAttentionName(Line, _handle);
        set => NativeBridgeApi.SetAttentionName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets per-attention metadata emitted by detailed engine inspection.
    /// 获取或设置由详细 engine inspector 输出的 attention 元数据。
    /// </summary>
    public string Metadata
    {
        get => NativeBridgeApi.GetAttentionMetadata(Line, _handle);
        set => NativeBridgeApi.SetAttentionMetadata(Line, _handle, value);
    }

    /// <summary>
    /// Gets the number of attention inputs.
    /// 获取 attention 输入数量。
    /// </summary>
    public int InputCount => NativeBridgeApi.GetAttentionInputCount(Line, _handle);

    /// <summary>
    /// Gets the number of attention outputs.
    /// 获取 attention 输出数量。
    /// </summary>
    public int OutputCount => NativeBridgeApi.GetAttentionOutputCount(Line, _handle);

    /// <summary>
    /// Sets the normalization operation used inside attention.
    /// 设置 attention 内部使用的归一化操作。
    /// </summary>
    public bool SetNormalizationOperation(TensorRtAttentionNormalizationOperation operation)
    {
        return NativeBridgeApi.SetAttentionNormalizationOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the normalization operation used inside attention.
    /// 获取 attention 内部使用的归一化操作。
    /// </summary>
    public TensorRtAttentionNormalizationOperation GetNormalizationOperation()
    {
        return NativeBridgeApi.GetAttentionNormalizationOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the optional attention mask tensor.
    /// 设置可选的 attention mask 张量。
    /// </summary>
    public bool SetMask(TensorRtTensor mask)
    {
        ValidateTensor(mask, nameof(mask));
        return NativeBridgeApi.SetAttentionMask(Line, _handle, mask.Handle);
    }

    /// <summary>
    /// Tries to get the optional attention mask tensor.
    /// 尝试获取可选的 attention mask 张量。
    /// </summary>
    public bool TryGetMask(out TensorRtTensor? mask)
    {
        SafeTensorRtObjectHandle? handle = NativeBridgeApi.GetAttentionMask(Line, _handle);
        if (handle == null)
        {
            mask = null;
            return false;
        }

        mask = new TensorRtTensor(Line, handle);
        return true;
    }

    /// <summary>
    /// Sets the causal mask alignment kind.
    /// 设置因果 mask 的对齐方向。
    /// </summary>
    public bool SetCausalKind(TensorRtCausalMaskKind causalKind)
    {
        return NativeBridgeApi.SetAttentionCausalKind(Line, _handle, causalKind);
    }

    /// <summary>
    /// Gets the causal mask alignment kind.
    /// 获取因果 mask 的对齐方向。
    /// </summary>
    public TensorRtCausalMaskKind GetCausalKind()
    {
        return NativeBridgeApi.GetAttentionCausalKind(Line, _handle);
    }

    /// <summary>
    /// Sets whether TensorRT may decompose attention into multiple kernels.
    /// 设置 TensorRT 是否可以将 attention 分解为多个 kernel。
    /// </summary>
    public bool SetDecomposable(bool decomposable)
    {
        return NativeBridgeApi.SetAttentionDecomposable(Line, _handle, decomposable);
    }

    /// <summary>
    /// Gets whether TensorRT may decompose attention into multiple kernels.
    /// 获取 TensorRT 是否可以将 attention 分解为多个 kernel。
    /// </summary>
    public bool GetDecomposable()
    {
        return NativeBridgeApi.GetAttentionDecomposable(Line, _handle);
    }

    /// <summary>
    /// Replaces one attention input tensor.
    /// 替换一个 attention 输入张量。
    /// </summary>
    public bool SetInput(int index, TensorRtTensor input)
    {
        ValidateTensor(input, nameof(input));
        return NativeBridgeApi.SetAttentionInput(Line, _handle, index, input.Handle);
    }

    /// <summary>
    /// Gets an attention input tensor by index.
    /// 按索引获取 attention 输入张量。
    /// </summary>
    public TensorRtTensor GetInput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetAttentionInput(Line, _handle, index));
    }

    /// <summary>
    /// Gets an attention output tensor by index.
    /// 按索引获取 attention 输出张量。
    /// </summary>
    public TensorRtTensor GetOutput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetAttentionOutput(Line, _handle, index));
    }

    /// <summary>
    /// Sets the normalization quantization scale tensor.
    /// 设置 normalization 输出量化 scale 张量。
    /// </summary>
    public bool SetNormalizationQuantizeScale(TensorRtTensor scale)
    {
        ValidateTensor(scale, nameof(scale));
        return NativeBridgeApi.SetAttentionNormalizationQuantizeScale(Line, _handle, scale.Handle);
    }

    /// <summary>
    /// Tries to get the normalization quantization scale tensor.
    /// 尝试获取 normalization 输出量化 scale 张量。
    /// </summary>
    public bool TryGetNormalizationQuantizeScale(out TensorRtTensor? scale)
    {
        SafeTensorRtObjectHandle? handle = NativeBridgeApi.GetAttentionNormalizationQuantizeScale(Line, _handle);
        if (handle == null)
        {
            scale = null;
            return false;
        }

        scale = new TensorRtTensor(Line, handle);
        return true;
    }

    /// <summary>
    /// Sets the data type used for quantized normalization output.
    /// 设置 normalization 量化输出使用的数据类型。
    /// </summary>
    public bool SetNormalizationQuantizeToType(TensorRtDataType dataType)
    {
        return NativeBridgeApi.SetAttentionNormalizationQuantizeToType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the data type used for quantized normalization output.
    /// 获取 normalization 量化输出使用的数据类型。
    /// </summary>
    public TensorRtDataType GetNormalizationQuantizeToType()
    {
        return NativeBridgeApi.GetAttentionNormalizationQuantizeToType(Line, _handle);
    }

    /// <summary>
    /// Sets the number of ranks for multi-device attention.
    /// 设置多设备 attention 的 rank 数量。
    /// </summary>
    public bool SetRankCount(int rankCount)
    {
        return NativeBridgeApi.SetAttentionRankCount(Line, _handle, rankCount);
    }

    /// <summary>
    /// Gets the number of ranks for multi-device attention.
    /// 获取多设备 attention 的 rank 数量。
    /// </summary>
    public int GetRankCount()
    {
        return NativeBridgeApi.GetAttentionRankCount(Line, _handle);
    }

    /// <summary>
    /// Sets the query tensor layout form.
    /// 设置 query 张量布局形式。
    /// </summary>
    public bool SetQueryForm(TensorRtAttentionIoForm form)
    {
        return NativeBridgeApi.SetAttentionQueryForm(Line, _handle, form);
    }

    /// <summary>
    /// Gets the query tensor layout form.
    /// 获取 query 张量布局形式。
    /// </summary>
    public TensorRtAttentionIoForm GetQueryForm()
    {
        return NativeBridgeApi.GetAttentionQueryForm(Line, _handle);
    }

    /// <summary>
    /// Sets the key/value tensor layout form.
    /// 设置 key/value 张量布局形式。
    /// </summary>
    public bool SetKeyValueForm(TensorRtAttentionIoForm form)
    {
        return NativeBridgeApi.SetAttentionKeyValueForm(Line, _handle, form);
    }

    /// <summary>
    /// Gets the key/value tensor layout form.
    /// 获取 key/value 张量布局形式。
    /// </summary>
    public TensorRtAttentionIoForm GetKeyValueForm()
    {
        return NativeBridgeApi.GetAttentionKeyValueForm(Line, _handle);
    }

    /// <summary>
    /// Sets the optional query lengths tensor.
    /// 设置可选的 query lengths 张量。
    /// </summary>
    public bool SetQueryLengths(TensorRtTensor lengths)
    {
        ValidateTensor(lengths, nameof(lengths));
        return NativeBridgeApi.SetAttentionQueryLengths(Line, _handle, lengths.Handle);
    }

    /// <summary>
    /// Tries to get the optional query lengths tensor.
    /// 尝试获取可选的 query lengths 张量。
    /// </summary>
    public bool TryGetQueryLengths(out TensorRtTensor? lengths)
    {
        SafeTensorRtObjectHandle? handle = NativeBridgeApi.GetAttentionQueryLengths(Line, _handle);
        if (handle == null)
        {
            lengths = null;
            return false;
        }

        lengths = new TensorRtTensor(Line, handle);
        return true;
    }

    /// <summary>
    /// Sets the optional key/value lengths tensor.
    /// 设置可选的 key/value lengths 张量。
    /// </summary>
    public bool SetKeyValueLengths(TensorRtTensor lengths)
    {
        ValidateTensor(lengths, nameof(lengths));
        return NativeBridgeApi.SetAttentionKeyValueLengths(Line, _handle, lengths.Handle);
    }

    /// <summary>
    /// Tries to get the optional key/value lengths tensor.
    /// 尝试获取可选的 key/value lengths 张量。
    /// </summary>
    public bool TryGetKeyValueLengths(out TensorRtTensor? lengths)
    {
        SafeTensorRtObjectHandle? handle = NativeBridgeApi.GetAttentionKeyValueLengths(Line, _handle);
        if (handle == null)
        {
            lengths = null;
            return false;
        }

        lengths = new TensorRtTensor(Line, handle);
        return true;
    }

    /// <summary>
    /// Releases the managed bridge reference for this network-owned attention.
    /// 释放此 network-owned attention 的托管桥接引用。
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
            throw new ArgumentException("Tensor must belong to the same TensorRT API line as the attention.", parameterName);
        }
    }
}
