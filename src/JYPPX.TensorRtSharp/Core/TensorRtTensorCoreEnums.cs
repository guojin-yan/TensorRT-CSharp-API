using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT TensorRtDataType values.
/// 表示 TensorRT TensorRtDataType 枚举值。
/// </summary>
public enum TensorRtDataType
{
    /// <summary>
    /// Represents the Float value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Float 取值。
    /// </summary>
    Float = 0,
    /// <summary>
    /// Represents the Half value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Half 取值。
    /// </summary>
    Half = 1,
    /// <summary>
    /// Represents the Int8 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int8 取值。
    /// </summary>
    Int8 = 2,
    /// <summary>
    /// Represents the Int32 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int32 取值。
    /// </summary>
    Int32 = 3,
    /// <summary>
    /// Represents the Bool value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Bool 取值。
    /// </summary>
    Bool = 4,
    /// <summary>
    /// Represents the UInt8 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 UInt8 取值。
    /// </summary>
    UInt8 = 5,
    /// <summary>
    /// Represents the Float8 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Float8 取值。
    /// </summary>
    Float8 = 6,
    /// <summary>
    /// Represents the BFloat16 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 BFloat16 取值。
    /// </summary>
    BFloat16 = 7,
    /// <summary>
    /// Represents the Int64 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int64 取值。
    /// </summary>
    Int64 = 8,
    /// <summary>
    /// Represents the Int4 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int4 取值。
    /// </summary>
    Int4 = 9,
    /// <summary>
    /// Represents the Float4 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Float4 取值。
    /// </summary>
    Float4 = 10,
    /// <summary>
    /// Represents the E8M0 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 E8M0 取值。
    /// </summary>
    E8M0 = 11,
    /// <summary>
    /// Represents the Unknown value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Unknown 取值。
    /// </summary>
    Unknown = 999
}
/// <summary>
/// Represents TensorRT TensorRtIOMode values.
/// 表示 TensorRT TensorRtIOMode 枚举值。
/// </summary>
public enum TensorRtIOMode
{
    /// <summary>
    /// Represents the Unknown value of TensorRtIOMode.
    /// 表示 TensorRtIOMode 的 Unknown 取值。
    /// </summary>
    Unknown = 0,
    /// <summary>
    /// Represents the Input value of TensorRtIOMode.
    /// 表示 TensorRtIOMode 的 Input 取值。
    /// </summary>
    Input = 1,
    /// <summary>
    /// Represents the Output value of TensorRtIOMode.
    /// 表示 TensorRtIOMode 的 Output 取值。
    /// </summary>
    Output = 2
}

/// <summary>
/// Represents TensorRT TensorRtTensorLocation values.
/// 表示 TensorRT TensorRtTensorLocation 枚举值。
/// </summary>
public enum TensorRtTensorLocation
{
    /// <summary>
    /// Represents the Device value of TensorRtTensorLocation.
    /// 表示 TensorRtTensorLocation 的 Device 取值。
    /// </summary>
    Device = 0,
    /// <summary>
    /// Represents the Host value of TensorRtTensorLocation.
    /// 表示 TensorRtTensorLocation 的 Host 取值。
    /// </summary>
    Host = 1
}
