using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// CUDA peer-to-peer attribute identifiers.
/// CUDA 点对点访问属性标识。
/// </summary>
public enum CudaDeviceP2PAttribute
{
    /// <summary>
    /// Relative performance rank for the link between two devices.
    /// 两个设备之间链路的相对性能等级。
    /// </summary>
    PerformanceRank = 1,

    /// <summary>
    /// Whether peer access is supported for the selected devices.
    /// 指定设备之间是否支持 peer access。
    /// </summary>
    AccessSupported = 2,

    /// <summary>
    /// Whether native atomic operations are supported over the link.
    /// 链路上是否支持 native atomic 操作。
    /// </summary>
    NativeAtomicSupported = 3,

    /// <summary>
    /// Whether CUDA array access is supported over the peer link.
    /// peer 链路上是否支持 CUDA array 访问。
    /// </summary>
    CudaArrayAccessSupported = 4
}
/// <summary>
/// CUDA atomic operations accepted by CUDA 13 host/P2P atomic capability queries.
/// CUDA 13 host/P2P atomic 能力查询支持的 atomic operation。
/// </summary>
public enum CudaAtomicOperation
{
    /// <summary>Integer add. 整数加法。</summary>
    IntegerAdd = 0,
    /// <summary>Integer minimum. 整数最小值。</summary>
    IntegerMin = 1,
    /// <summary>Integer maximum. 整数最大值。</summary>
    IntegerMax = 2,
    /// <summary>Integer increment. 整数递增。</summary>
    IntegerIncrement = 3,
    /// <summary>Integer decrement. 整数递减。</summary>
    IntegerDecrement = 4,
    /// <summary>Bitwise and. 按位与。</summary>
    And = 5,
    /// <summary>Bitwise or. 按位或。</summary>
    Or = 6,
    /// <summary>Bitwise xor. 按位异或。</summary>
    Xor = 7,
    /// <summary>Exchange. 交换。</summary>
    Exchange = 8,
    /// <summary>Compare and swap. 比较并交换。</summary>
    CompareAndSwap = 9,
    /// <summary>Floating-point add. 浮点加法。</summary>
    FloatAdd = 10,
    /// <summary>Floating-point minimum. 浮点最小值。</summary>
    FloatMin = 11,
    /// <summary>Floating-point maximum. 浮点最大值。</summary>
    FloatMax = 12
}

/// <summary>
/// Capability bitmask returned for a CUDA atomic operation.
/// CUDA atomic operation 返回的能力位掩码。
/// </summary>
[Flags]
public enum CudaAtomicCapability : uint
{
    /// <summary>No native capability was reported. 未报告原生能力。</summary>
    None = 0,
    /// <summary>Signed operand support. 支持有符号操作数。</summary>
    Signed = 1u << 0,
    /// <summary>Unsigned operand support. 支持无符号操作数。</summary>
    Unsigned = 1u << 1,
    /// <summary>Reduction support. 支持 reduction。</summary>
    Reduction = 1u << 2,
    /// <summary>32-bit scalar support. 支持 32-bit scalar。</summary>
    Scalar32 = 1u << 3,
    /// <summary>64-bit scalar support. 支持 64-bit scalar。</summary>
    Scalar64 = 1u << 4,
    /// <summary>128-bit scalar support. 支持 128-bit scalar。</summary>
    Scalar128 = 1u << 5,
    /// <summary>Four-lane 32-bit vector support. 支持 Vector32x4。</summary>
    Vector32x4 = 1u << 6
}
