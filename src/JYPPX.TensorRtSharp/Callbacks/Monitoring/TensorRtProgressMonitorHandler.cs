namespace JYPPX.TensorRtSharp;

/// <summary>
/// Receives TensorRT progress monitor events copied across the native boundary.
/// 接收从 native 边界复制过来的 TensorRT progress monitor 事件。
/// </summary>
/// <param name="progressEvent">The copied progress event. 复制后的 progress 事件。</param>
/// <returns>
/// <see langword="true"/> to continue a build after <see cref="TensorRtProgressMonitorEventKind.StepComplete"/>;
/// <see langword="false"/> to request cancellation. Phase-start and phase-finish return values are ignored.
/// 对 <see cref="TensorRtProgressMonitorEventKind.StepComplete"/> 返回 <see langword="true"/> 表示继续构建，返回
/// <see langword="false"/> 表示请求取消。phase-start 和 phase-finish 的返回值会被忽略。
/// </returns>
public delegate bool TensorRtProgressMonitorHandler(TensorRtProgressMonitorEvent progressEvent);
