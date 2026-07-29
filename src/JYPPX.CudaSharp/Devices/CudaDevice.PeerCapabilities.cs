using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Gets whether one CUDA device can access another through peer access.
    /// 获取一个 CUDA 设备是否可以通过 peer access 访问另一个设备。
    /// </summary>
    /// <param name="ordinal">The source CUDA device ordinal. 源 CUDA 设备序号。</param>
    /// <param name="peerOrdinal">The peer CUDA device ordinal. 对端 CUDA 设备序号。</param>
    /// <returns><see langword="true"/> when peer access is available. 可以进行 peer access 时返回 <see langword="true"/>。</returns>
    public static bool CanAccessPeer(int ordinal, int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.CanDeviceAccessPeer(ordinal, peerOrdinal);
    }

    /// <summary>
    /// Gets a CUDA peer-to-peer attribute for a pair of devices.
    /// 获取两个 CUDA 设备之间的 peer-to-peer 属性。
    /// </summary>
    /// <param name="attribute">The CUDA P2P attribute to query. 要查询的 CUDA P2P 属性。</param>
    /// <param name="sourceOrdinal">The source CUDA device ordinal. 源 CUDA 设备序号。</param>
    /// <param name="destinationOrdinal">The destination CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <returns>The raw CUDA attribute value. CUDA 返回的原始属性值。</returns>
    public static int GetP2PAttribute(CudaDeviceP2PAttribute attribute, int sourceOrdinal, int destinationOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceP2PAttribute((int)attribute, sourceOrdinal, destinationOrdinal);
    }

    /// <summary>
    /// Gets native host atomic capabilities for the selected CUDA operations on a device.
    /// 获取指定设备对一组 CUDA atomic operation 的 host atomic 原生能力。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="operations">The CUDA atomic operations to query. 要查询的 CUDA atomic operation 列表。</param>
    /// <returns>One capability bitmask per operation, in the same order. 按输入顺序返回每个 operation 的能力位掩码。</returns>
    public static CudaAtomicCapability[] GetHostAtomicCapabilities(int ordinal, IReadOnlyList<CudaAtomicOperation> operations)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceHostAtomicCapabilities(ordinal, CopyAtomicOperations(operations));
    }

    /// <summary>
    /// Gets native peer-to-peer atomic capabilities for the selected CUDA operations between two devices.
    /// 获取两个 CUDA 设备之间对一组 CUDA atomic operation 的 P2P atomic 原生能力。
    /// </summary>
    /// <param name="sourceOrdinal">The source CUDA device ordinal. 源 CUDA 设备序号。</param>
    /// <param name="destinationOrdinal">The destination CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="operations">The CUDA atomic operations to query. 要查询的 CUDA atomic operation 列表。</param>
    /// <returns>One capability bitmask per operation, in the same order. 按输入顺序返回每个 operation 的能力位掩码。</returns>
    public static CudaAtomicCapability[] GetP2PAtomicCapabilities(int sourceOrdinal, int destinationOrdinal, IReadOnlyList<CudaAtomicOperation> operations)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceP2PAtomicCapabilities(sourceOrdinal, destinationOrdinal, CopyAtomicOperations(operations));
    }

    /// <summary>
    /// Enables peer access from the current device to the specified peer device.
    /// 启用当前设备到指定 peer 设备的 peer access。
    /// </summary>
    /// <param name="peerOrdinal">The peer CUDA device ordinal. 对端 CUDA 设备序号。</param>
    public static void EnablePeerAccess(int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.EnablePeerAccess(peerOrdinal, 0);
    }

    /// <summary>
    /// Disables peer access from the current device to the specified peer device.
    /// 关闭当前设备到指定 peer 设备的 peer access。
    /// </summary>
    /// <param name="peerOrdinal">The peer CUDA device ordinal. 对端 CUDA 设备序号。</param>
    public static void DisablePeerAccess(int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.DisablePeerAccess(peerOrdinal);
    }

    private static CudaAtomicOperation[] CopyAtomicOperations(IReadOnlyList<CudaAtomicOperation> operations)
    {
        if (operations == null)
        {
            throw new ArgumentNullException(nameof(operations));
        }

        if (operations.Count == 0)
        {
            throw new ArgumentException("At least one CUDA atomic operation is required.", nameof(operations));
        }

        CudaAtomicOperation[] copy = new CudaAtomicOperation[operations.Count];
        for (int index = 0; index < operations.Count; index++)
        {
            copy[index] = operations[index];
        }

        return copy;
    }

}
