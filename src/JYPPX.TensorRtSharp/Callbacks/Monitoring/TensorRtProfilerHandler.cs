namespace JYPPX.TensorRtSharp;

/// <summary>
/// Receives TensorRT layer profiling records copied across the native boundary.
/// 接收从 native 边界复制过来的 TensorRT layer profiling 记录。
/// </summary>
/// <param name="layerName">The layer name reported by TensorRT. TensorRT 报告的 layer 名称。</param>
/// <param name="milliseconds">The layer execution time in milliseconds. layer 执行耗时，单位毫秒。</param>
public delegate void TensorRtProfilerHandler(string layerName, float milliseconds);
