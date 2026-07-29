using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Selects TensorRT 10.x cross-platform runtime target.
/// 选择 TensorRT 10.x 跨平台 runtime 目标。
/// </summary>
public enum TensorRtRuntimePlatform
{
    /// <summary>
    /// Engine can run only on the same platform family it was built for.
    /// Engine 仅面向构建时相同的平台家族运行。
    /// </summary>
    SameAsBuild = 0,

    /// <summary>
    /// TensorRT 10.x Windows AMD64 target. NVIDIA currently documents this as a Linux-build-to-Windows-target mode.
    /// TensorRT 10.x Windows AMD64 目标；NVIDIA 当前文档将其定义为 Linux 构建面向 Windows 运行的模式。
    /// </summary>
    WindowsAmd64 = 1
}
