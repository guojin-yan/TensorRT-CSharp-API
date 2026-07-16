namespace JYPPX.CudaSharp;

/// <summary>
/// Describes the result of a CUDA executable-graph update attempt.
/// 描述 CUDA executable graph 更新尝试的结果。
/// </summary>
public enum CudaGraphExecUpdateResult
{
    Success = 0,
    Error = 1,
    TopologyChanged = 2,
    NodeTypeChanged = 3,
    FunctionChanged = 4,
    ParametersChanged = 5,
    NotSupported = 6,
    UnsupportedFunctionChange = 7,
    AttributesChanged = 8
}
