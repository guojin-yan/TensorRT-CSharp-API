using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 数据类型枚举，用于表示权重和张量的类型
    /// Data type enumeration for representing the type of weights and tensors
    /// </summary>
    /// <remarks>
    /// 此枚举定义了TensorRT支持的各种数据类型，包括浮点数、整数和布尔值等
    /// This enumeration defines various data types supported by TensorRT, including floating-point numbers, integers, and booleans
    /// </remarks>
    public enum TrtDataType : int
    {
        /// <summary>
        /// 32位浮点数格式
        /// 32-bit floating point format
        /// </summary>
        kFLOAT = 0,

        /// <summary>
        /// IEEE 16位浮点数格式 -- 具有5位指数和11位尾数
        /// IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand
        /// </summary>
        kHALF = 1,

        /// <summary>
        /// 有符号8位整数，表示量化的浮点值
        /// Signed 8-bit integer representing a quantized floating-point value
        /// </summary>
        kINT8 = 2,

        /// <summary>
        /// 有符号32位整数格式
        /// Signed 32-bit integer format
        /// </summary>
        kINT32 = 3,

        /// <summary>
        /// 8位布尔值。0 = false，1 = true，其他值未定义
        /// 8-bit boolean. 0 = false, 1 = true, other values undefined
        /// </summary>
        kBOOL = 4,

        /// <summary>
        /// 无符号8位整数格式
        /// Unsigned 8-bit integer format
        /// </summary>
        /// <remarks>
        /// 不能用于表示量化的浮点值。在与其他TensorRT层一起使用之前，使用IdentityLayer将网络级kUINT8输入转换为{kFLOAT, kHALF}，
        /// 或在kUINT8网络级输出之前将中间输出从{kFLOAT, kHALF}转换为kUINT8。
        /// kUINT8转换仅支持{kFLOAT, kHALF}。kUINT8到{kFLOAT, kHALF}的转换会将整数值转换为等效的浮点值。
        /// {kFLOAT, kHALF}到kUINT8的转换将通过向零截断将浮点值转换为整数值。
        /// 对于截断后范围不在[0.0F, 256.0F)内的浮点值，此转换具有未定义的行为。
        /// kUINT8转换不支持{kINT8, kINT32, kBOOL}。
        /// Cannot be used to represent quantized floating-point values.
        /// Use the IdentityLayer to convert kUINT8 network-level inputs to {kFLOAT, kHALF} prior
        /// to use with other TensorRT layers, or to convert intermediate output
        /// before kUINT8 network-level outputs from {kFLOAT, kHALF} to kUINT8.
        /// kUINT8 conversions are only supported for {kFLOAT, kHALF}.
        /// kUINT8 to {kFLOAT, kHALF} conversion will convert the integer values
        /// to equivalent floating point values.
        /// {kFLOAT, kHALF} to kUINT8 conversion will convert the floating point values
        /// to integer values by truncating towards zero. This conversion has undefined behavior for
        /// floating point values outside the range [0.0F, 256.0F) after truncation.
        /// kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
        /// </remarks>
        kUINT8 = 5,

        /// <summary>
        /// 8位浮点类型，带有1个符号位，4个指数位，3个尾数位，以及指数偏移7
        /// Signed 8-bit floating point with
        /// 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7
        /// </summary>
        kFP8 = 6,

        /// <summary>
        /// Brain浮点数 -- 具有8位指数和8位尾数
        /// Brain float -- has an 8 bit exponent and 8 bit significand
        /// </summary>
        kBF16 = 7,

        /// <summary>
        /// 有符号64位整数类型
        /// Signed 64-bit integer type
        /// </summary>
        kINT64 = 8,

        /// <summary>
        /// 有符号4位整数类型
        /// Signed 4-bit integer type
        /// </summary>
        kINT4 = 9,

        /// <summary>
        /// 4位浮点数类型
        /// 4-bit floating point type
        /// </summary>
        /// <remarks>
        /// 1位符号，2位指数，1位尾数
        /// 1 bit sign, 2 bit exponent, 1 bit mantissa
        /// </remarks>
        kFP4 = 10,

        /// <summary>
        /// 用于量化比例的无符号指数仅8位浮点类型表示
        /// Unsigned representation of exponent-only 8-bit floating point type for quantization scales
        /// </summary>
        kE8M0 = 11,
    }

}
