namespace JYPPX.CudaSharp;

/// <summary>
/// Describes the result of a CUDA executable-graph update attempt.
/// 描述 CUDA executable graph 更新尝试的结果。
/// </summary>
public enum CudaGraphExecUpdateResult
{
    /// <summary>The executable graph was updated successfully. executable graph 已成功更新。</summary>
    Success = 0,
    /// <summary>The update failed with a general error. 更新因一般错误而失败。</summary>
    Error = 1,
    /// <summary>The graph topology changed incompatibly. graph 拓扑发生了不兼容变化。</summary>
    TopologyChanged = 2,
    /// <summary>A corresponding node changed type. 对应节点的类型发生了变化。</summary>
    NodeTypeChanged = 3,
    /// <summary>A node function changed incompatibly. 节点函数发生了不兼容变化。</summary>
    FunctionChanged = 4,
    /// <summary>Node parameters changed incompatibly. 节点参数发生了不兼容变化。</summary>
    ParametersChanged = 5,
    /// <summary>The requested update is not supported. 不支持所请求的更新。</summary>
    NotSupported = 6,
    /// <summary>The function change is specifically unsupported. 特定函数变更不受支持。</summary>
    UnsupportedFunctionChange = 7,
    /// <summary>Node attributes changed incompatibly. 节点属性发生了不兼容变化。</summary>
    AttributesChanged = 8
}
