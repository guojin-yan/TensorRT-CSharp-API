using System;

namespace JYPPX.CudaSharp;

/// <summary>Identifies a copied scalar or owner-bound device-memory kernel argument. 标识复制型标量或 owner-bound device-memory kernel 参数。</summary>
public enum CudaKernelArgumentKind
{
    /// <summary>A copied scalar value. 复制型标量值。</summary>
    Scalar = 1,
    /// <summary>A CUDA device-memory owner and byte offset. CUDA device-memory owner 与字节偏移。</summary>
    DeviceMemory = 2
}

/// <summary>Identifies the managed scalar layout copied into a CUDA kernel argument. 标识复制到 CUDA kernel 参数的托管标量布局。</summary>
public enum CudaKernelScalarType
{
    /// <summary>No scalar is present. 不包含标量。</summary>
    None = 0,
    /// <summary>A one-byte Boolean value. 单字节 Boolean 值。</summary>
    Boolean = 1,
    /// <summary>An unsigned 8-bit integer. 无符号 8 位整数。</summary>
    Byte = 2,
    /// <summary>A signed 8-bit integer. 有符号 8 位整数。</summary>
    SByte = 3,
    /// <summary>A signed 16-bit integer. 有符号 16 位整数。</summary>
    Int16 = 4,
    /// <summary>An unsigned 16-bit integer. 无符号 16 位整数。</summary>
    UInt16 = 5,
    /// <summary>A signed 32-bit integer. 有符号 32 位整数。</summary>
    Int32 = 6,
    /// <summary>An unsigned 32-bit integer. 无符号 32 位整数。</summary>
    UInt32 = 7,
    /// <summary>A signed 64-bit integer. 有符号 64 位整数。</summary>
    Int64 = 8,
    /// <summary>An unsigned 64-bit integer. 无符号 64 位整数。</summary>
    UInt64 = 9,
    /// <summary>An IEEE 754 single-precision value. IEEE 754 单精度值。</summary>
    Single = 10,
    /// <summary>An IEEE 754 double-precision value. IEEE 754 双精度值。</summary>
    Double = 11
}

/// <summary>
/// Stores a copied scalar or a managed CUDA memory owner without exposing native pointers.
/// 保存复制型标量或托管 CUDA memory owner，不暴露 native pointer。
/// </summary>
public sealed class CudaKernelArgument
{
    private readonly byte[]? _scalarBytes;
    private readonly CudaMemory? _memory;

    private CudaKernelArgument(CudaKernelScalarType scalarType, byte[] scalarBytes)
    {
        Kind = CudaKernelArgumentKind.Scalar;
        ScalarType = scalarType;
        _scalarBytes = scalarBytes;
    }

    private CudaKernelArgument(CudaMemory memory, int memoryOffset)
    {
        Kind = CudaKernelArgumentKind.DeviceMemory;
        ScalarType = CudaKernelScalarType.None;
        _memory = memory;
        MemoryOffset = memoryOffset;
    }

    /// <summary>Gets the argument ownership/layout kind. 获取参数 ownership/layout 类型。</summary>
    public CudaKernelArgumentKind Kind { get; }

    /// <summary>Gets the scalar type, or <see cref="CudaKernelScalarType.None"/> for device memory. 获取标量类型；device memory 返回 None。</summary>
    public CudaKernelScalarType ScalarType { get; }

    /// <summary>Gets the copied scalar size. 获取复制型标量大小。</summary>
    public int ScalarSizeInBytes => _scalarBytes?.Length ?? 0;

    /// <summary>Gets the byte offset within a device-memory owner. 获取 device-memory owner 内的字节偏移。</summary>
    public int MemoryOffset { get; }

    /// <summary>Copies a Boolean scalar argument. 复制 Boolean 标量参数。</summary>
    public static CudaKernelArgument FromBoolean(bool value) => Scalar(CudaKernelScalarType.Boolean, BitConverter.GetBytes(value));

    /// <summary>Copies an unsigned 8-bit scalar argument. 复制无符号 8 位标量参数。</summary>
    public static CudaKernelArgument FromByte(byte value) => Scalar(CudaKernelScalarType.Byte, new[] { value });

    /// <summary>Copies a signed 8-bit scalar argument. 复制有符号 8 位标量参数。</summary>
    public static CudaKernelArgument FromSByte(sbyte value) => Scalar(CudaKernelScalarType.SByte, new[] { unchecked((byte)value) });

    /// <summary>Copies a signed 16-bit scalar argument. 复制有符号 16 位标量参数。</summary>
    public static CudaKernelArgument FromInt16(short value) => Scalar(CudaKernelScalarType.Int16, BitConverter.GetBytes(value));

    /// <summary>Copies an unsigned 16-bit scalar argument. 复制无符号 16 位标量参数。</summary>
    public static CudaKernelArgument FromUInt16(ushort value) => Scalar(CudaKernelScalarType.UInt16, BitConverter.GetBytes(value));

    /// <summary>Copies a signed 32-bit scalar argument. 复制有符号 32 位标量参数。</summary>
    public static CudaKernelArgument FromInt32(int value) => Scalar(CudaKernelScalarType.Int32, BitConverter.GetBytes(value));

    /// <summary>Copies an unsigned 32-bit scalar argument. 复制无符号 32 位标量参数。</summary>
    public static CudaKernelArgument FromUInt32(uint value) => Scalar(CudaKernelScalarType.UInt32, BitConverter.GetBytes(value));

    /// <summary>Copies a signed 64-bit scalar argument. 复制有符号 64 位标量参数。</summary>
    public static CudaKernelArgument FromInt64(long value) => Scalar(CudaKernelScalarType.Int64, BitConverter.GetBytes(value));

    /// <summary>Copies an unsigned 64-bit scalar argument. 复制无符号 64 位标量参数。</summary>
    public static CudaKernelArgument FromUInt64(ulong value) => Scalar(CudaKernelScalarType.UInt64, BitConverter.GetBytes(value));

    /// <summary>Copies an IEEE 754 single-precision scalar argument. 复制 IEEE 754 单精度标量参数。</summary>
    public static CudaKernelArgument FromSingle(float value) => Scalar(CudaKernelScalarType.Single, BitConverter.GetBytes(value));

    /// <summary>Copies an IEEE 754 double-precision scalar argument. 复制 IEEE 754 双精度标量参数。</summary>
    public static CudaKernelArgument FromDouble(double value) => Scalar(CudaKernelScalarType.Double, BitConverter.GetBytes(value));

    /// <summary>Creates an owner-bound device-memory argument. 创建 owner-bound device-memory 参数。</summary>
    public static CudaKernelArgument FromDeviceMemory(CudaMemory memory, int byteOffset = 0)
    {
        if (memory == null) throw new ArgumentNullException(nameof(memory));
        if (memory.Handle.IsClosed || memory.Handle.IsInvalid) throw new ObjectDisposedException(nameof(memory));
        if (byteOffset < 0 || byteOffset >= memory.SizeInBytes) throw new ArgumentOutOfRangeException(nameof(byteOffset));
        return new CudaKernelArgument(memory, byteOffset);
    }

    internal byte[] ScalarBytes => _scalarBytes ?? throw new InvalidOperationException("The kernel argument is not a scalar.");
    internal CudaMemory Memory => _memory ?? throw new InvalidOperationException("The kernel argument is not device memory.");

    private static CudaKernelArgument Scalar(CudaKernelScalarType scalarType, byte[] bytes)
    {
        return new CudaKernelArgument(scalarType, bytes);
    }
}
