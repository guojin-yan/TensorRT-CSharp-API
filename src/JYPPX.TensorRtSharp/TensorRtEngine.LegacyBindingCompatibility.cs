namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets the engine binding count using the TensorRT 8 legacy binding vocabulary.
    /// 使用 TensorRT 8 legacy binding 语义获取 engine binding 数量。
    /// </summary>
    /// <remarks>
    /// On TensorRT 10 and 11 this maps to the name-based I/O tensor count exposed by the bridge.
    /// 在 TensorRT 10 和 11 中，该属性映射到桥接层公开的基于名称的 I/O tensor 数量。
    /// </remarks>
    public int CompatibilityBindingCount => IOTensorCount;

    /// <summary>
    /// Gets a legacy-binding-compatible tensor snapshot for an engine binding index.
    /// 根据 engine binding 索引获取兼容 legacy binding 语义的 tensor 快照。
    /// </summary>
    /// <param name="bindingIndex">The zero-based binding index. 从零开始的 binding 索引。</param>
    /// <returns>A safe managed tensor metadata snapshot. 安全的托管 tensor 元数据快照。</returns>
    public TensorRtTensorInfo GetCompatibilityBindingInfo(int bindingIndex)
    {
        return GetIOTensorInfo(bindingIndex);
    }

    /// <summary>
    /// Gets a legacy-binding-compatible tensor name for an engine binding index.
    /// 根据 engine binding 索引获取兼容 legacy binding 语义的 tensor 名称。
    /// </summary>
    /// <param name="bindingIndex">The zero-based binding index. 从零开始的 binding 索引。</param>
    /// <returns>The tensor or binding name reported by TensorRT. TensorRT 报告的 tensor 或 binding 名称。</returns>
    public string GetCompatibilityBindingName(int bindingIndex)
    {
        return GetIOTensorName(bindingIndex);
    }

    /// <summary>
    /// Gets a legacy-binding-compatible I/O mode for an engine binding index.
    /// 根据 engine binding 索引获取兼容 legacy binding 语义的输入输出模式。
    /// </summary>
    /// <param name="bindingIndex">The zero-based binding index. 从零开始的 binding 索引。</param>
    /// <returns>The input/output mode for the binding. 该 binding 的输入输出模式。</returns>
    public TensorRtIOMode GetCompatibilityBindingIOMode(int bindingIndex)
    {
        return GetTensorIOMode(GetCompatibilityBindingName(bindingIndex));
    }

    /// <summary>
    /// Gets a legacy-binding-compatible shape for an engine binding index.
    /// 根据 engine binding 索引获取兼容 legacy binding 语义的形状。
    /// </summary>
    /// <param name="bindingIndex">The zero-based binding index. 从零开始的 binding 索引。</param>
    /// <returns>The static or profile-independent shape reported by TensorRT. TensorRT 报告的静态或 profile-independent 形状。</returns>
    public TensorRtDims GetCompatibilityBindingShape(int bindingIndex)
    {
        return GetTensorShape(GetCompatibilityBindingName(bindingIndex));
    }
}
