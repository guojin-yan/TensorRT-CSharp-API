namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes a TensorRT layer/role pair that can be refitted or is still missing.
/// 描述一个可 refit 或仍缺失的 TensorRT layer/role 组合。
/// </summary>
public readonly struct TensorRtRefitEntry
{
    /// <summary>
    /// Initializes a refit entry.
    /// 初始化 refit 条目。
    /// </summary>
    /// <param name="layerName">The TensorRT layer name. TensorRT 层名称。</param>
    /// <param name="role">The weight role for the layer. 该层对应的权重角色。</param>
    public TensorRtRefitEntry(string layerName, TensorRtWeightsRole role)
    {
        LayerName = layerName ?? string.Empty;
        Role = role;
    }

    /// <summary>
    /// Gets the TensorRT layer name.
    /// 获取 TensorRT 层名称。
    /// </summary>
    public string LayerName { get; }

    /// <summary>
    /// Gets the weight role that TensorRT expects for the layer.
    /// 获取 TensorRT 期望该层使用的权重角色。
    /// </summary>
    public TensorRtWeightsRole Role { get; }

    /// <summary>
    /// Returns a readable layer/role pair.
    /// 返回可读的 layer/role 组合。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{LayerName}:{Role}";
    }
}
