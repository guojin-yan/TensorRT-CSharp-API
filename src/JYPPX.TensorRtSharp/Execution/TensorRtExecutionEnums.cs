using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Selects how TensorRT 11 allocates execution-context device memory.
/// 选择 TensorRT 11 execution context 的设备内存分配策略。
/// </summary>
public enum TensorRtExecutionContextAllocationStrategy
{
    /// <summary>
    /// Allocate statically for the maximum requirement across profiles.
    /// 按所有 profile 中的最大需求静态分配。
    /// </summary>
    Static = 0,

    /// <summary>
    /// Reallocate when the active optimization profile changes.
    /// 当 active optimization profile 改变时重新分配。
    /// </summary>
    OnProfileChange = 1,

    /// <summary>
    /// The application provides device memory explicitly.
    /// 由应用程序显式提供 device memory。
    /// </summary>
    UserManaged = 2
}
