using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Initializes the primary CUDA context for a device using CUDA 12.0+ <c>cudaInitDevice</c>.
    /// 使用 CUDA 12.0+ <c>cudaInitDevice</c> 初始化指定设备的 primary context。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="deviceFlags">The CUDA runtime device flags to apply during initialization. 初始化时应用的 CUDA runtime device flags。</param>
    /// <param name="flags">Reserved CUDA flags. CUDA 保留 flags，通常为 0。</param>
    /// <remarks>
    /// This wrapper keeps the boundary scalar-only and does not expose CUDA context handles.
    /// 该封装仅暴露标量边界，不向托管层暴露 CUDA context 句柄。
    /// </remarks>
    public static void InitDevice(int ordinal, CudaDeviceRuntimeFlags deviceFlags = CudaDeviceRuntimeFlags.ScheduleAuto, uint flags = 0)
    {
        if (ordinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "CUDA device ordinal must be greater than or equal to zero.");
        }

        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.InitDevice(ordinal, (uint)deviceFlags, flags);
    }

    /// <summary>
    /// Gets an owner-safe bridge wrapper for a device primary execution context on CUDA 13.0 or later.
    /// 在 CUDA 13.0 或更高版本上获取设备主执行上下文的 owner-safe bridge 包装器。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>A non-destroying primary execution-context wrapper. 不会销毁主上下文的执行上下文包装器。</returns>
    public static CudaPrimaryExecutionContext GetPrimaryExecutionContext(int ordinal)
    {
        if (ordinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "CUDA device ordinal must be greater than or equal to zero.");
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaPrimaryExecutionContext(NativeCudaApi.GetPrimaryExecutionContext(ordinal));
    }

    /// <summary>
    /// Restricts CUDA runtime initialization to a caller-owned list of valid device ordinals using <c>cudaSetValidDevices</c>.
    /// 使用 <c>cudaSetValidDevices</c> 和 caller-owned 的设备序号列表限制 CUDA runtime 可初始化的设备集合。
    /// </summary>
    /// <param name="ordinals">The CUDA device ordinals that may be used by the process. 允许当前进程使用的 CUDA 设备序号。</param>
    /// <remarks>
    /// This method copies the managed list before calling native code. Call it before creating a CUDA context.
    /// 该方法会先复制托管列表再调用 native；应在创建 CUDA context 前调用。
    /// </remarks>
    public static void SetValidDevices(IReadOnlyList<int> ordinals)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetValidDevices(CopyDeviceOrdinals(ordinals));
    }

    /// <summary>
    /// Gets the PCI bus id string for a CUDA device.
    /// 获取 CUDA 设备的 PCI bus id 字符串。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The PCI bus id reported by CUDA, such as <c>0000:65:00.0</c>. CUDA 返回的 PCI bus id。</returns>
    public static string GetPciBusId(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDevicePciBusId(ordinal);
    }

    /// <summary>
    /// Resolves a CUDA device ordinal from a PCI bus id string.
    /// 根据 PCI bus id 字符串解析 CUDA 设备序号。
    /// </summary>
    /// <param name="pciBusId">The PCI bus id reported by CUDA. CUDA 返回的 PCI bus id。</param>
    /// <returns>The CUDA device ordinal. CUDA 设备序号。</returns>
    public static int GetByPciBusId(string pciBusId)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceByPciBusId(pciBusId);
    }

    /// <summary>
    /// Gets a CUDA device attribute as a boolean capability.
    /// 将 CUDA 设备属性读取为布尔能力值。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="attribute">The device attribute to query. 要查询的设备属性。</param>
    /// <returns><see langword="true"/> when the attribute is non-zero. 属性非零时返回 <see langword="true"/>。</returns>
    public static bool GetBooleanAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        return GetAttribute(ordinal, attribute) != 0;
    }

    /// <summary>
    /// Tries to read a CUDA device attribute without failing the whole capability snapshot.
    /// 尝试读取 CUDA device attribute，不会因为单个 attribute 不可用而中断整个能力快照。
    /// </summary>
    public static int? TryGetAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        try
        {
            return GetAttribute(ordinal, attribute);
        }
        catch (CudaException)
        {
            return null;
        }
    }

    /// <summary>
    /// Tries to read a CUDA device attribute as a boolean capability.
    /// 尝试把 CUDA device attribute 读取为布尔能力。
    /// </summary>
    public static bool? TryGetBooleanAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        int? value = TryGetAttribute(ordinal, attribute);
        return value.HasValue ? value.Value != 0 : (bool?)null;
    }

    /// <summary>
    /// Chooses the CUDA device that best matches a safe managed subset of <c>cudaDeviceProp</c> requirements.
    /// 根据托管层安全表达的 <c>cudaDeviceProp</c> 子集选择最匹配的 CUDA 设备。
    /// </summary>
    /// <param name="requirements">The desired CUDA device requirements. 期望的 CUDA 设备约束。</param>
    /// <returns>The CUDA device ordinal chosen by <c>cudaChooseDevice</c>. <c>cudaChooseDevice</c> 选择的 CUDA 设备序号。</returns>
    /// <remarks>
    /// The wrapper does not expose native <c>cudaDeviceProp</c> pointers. It copies a small caller-owned value structure into native code.
    /// 该封装不会暴露原生 <c>cudaDeviceProp</c> 指针，而是把小型 caller-owned 值结构复制到 native 侧。
    /// </remarks>
    public static int ChooseDevice(CudaDeviceSelectionRequirements requirements)
    {
        if (requirements == null)
        {
            throw new ArgumentNullException(nameof(requirements));
        }

        NativeBridgeLoader.EnsureInitialized();
        NativeCudaDeviceSelectionRequirements nativeRequirements = requirements.ToNative();
        return NativeCudaApi.ChooseDevice(in nativeRequirements);
    }

    private static int[] CopyDeviceOrdinals(IReadOnlyList<int> ordinals)
    {
        if (ordinals == null)
        {
            throw new ArgumentNullException(nameof(ordinals));
        }

        if (ordinals.Count == 0)
        {
            throw new ArgumentException("At least one CUDA device ordinal is required.", nameof(ordinals));
        }

        int[] copy = new int[ordinals.Count];
        for (int index = 0; index < ordinals.Count; index++)
        {
            int ordinal = ordinals[index];
            if (ordinal < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(ordinals), ordinal, "CUDA device ordinal must be greater than or equal to zero.");
            }

            copy[index] = ordinal;
        }

        return copy;
    }
}
