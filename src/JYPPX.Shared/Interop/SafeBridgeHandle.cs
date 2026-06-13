using Microsoft.Win32.SafeHandles;

namespace JYPPX.Shared.Interop;

/// <summary>
/// Common base type for native bridge SafeHandle implementations.
/// 原生 bridge `SafeHandle` 实现的公共基类。
/// </summary>
public abstract class SafeBridgeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    /// <summary>
    /// Initializes a bridge safe handle that owns the native resource.
    /// 初始化拥有原生资源所有权的 bridge safe handle。
    /// </summary>
    protected SafeBridgeHandle()
        : base(true)
    {
    }
}
