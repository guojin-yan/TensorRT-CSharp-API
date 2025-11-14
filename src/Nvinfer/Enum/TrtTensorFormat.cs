using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// TensorRT张量格式枚举，定义了不同的内存布局方式
    /// TensorRT tensor format enumeration defining different memory layout methods
    /// </summary>
    public enum TrtTensorFormat : int
    {
        /// <summary>
        /// 类似于C或C++中的数组的内存布局
        /// Memory layout is similar to an array in C or C++
        /// </summary>
        /// <remarks>
        /// 每个维度的步幅是其后维度的乘积，最后一个维度具有单位步幅
        /// 此格式支持所有TensorRT类型，对于DLA使用，张量大小限制在C,H,W范围内[1,8192]
        /// The stride of each dimension is the product of the dimensions after it.
        //! The last dimension has unit stride.
        //!
        //! This format supports all TensorRT types.
        //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
        /// </remarks>
        kLINEAR = 0,

        /// <summary>
        /// 主向量格式，每个向量有两个标量
        /// Vector-major format with two scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第三个
        /// 此格式需要FP16或BF16，并且至少有三个维度
        /// The vector dimension is third to last.
        //!
        //! This format requires FP16 or BF16 and at least three dimensions.
        /// </remarks>
        kCHW2 = 1,

        /// <summary>
        /// 从向量格式，每个向量有八个标量
        /// Vector-minor format with eight scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第三个
        /// 此格式需要FP16或BF16，并且至少有三个维度
        /// The vector dimension is third to last.
        //! This format requires FP16 or BF16 and at least three dimensions.
        /// </remarks>
        kHWC8 = 2,

        /// <summary>
        /// 主向量格式，每个向量有四个标量
        /// Vector-major format with four scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第三个
        /// 此格式需要INT8，并且至少有三个维度
        /// 对于INT8，向量维度的长度必须是编译时常量
        /// 已弃用：如果在DLA上运行，可以使用此格式进行加速，但需要注意C必须小于或等于4
        /// 如果用作DLA输入且未指定kGPU_FALLBACK构建选项，则需要满足DLA格式的行步幅要求
        /// The vector dimension is third to last.
        //!
        //! This format requires INT8 and at least three dimensions.
        //! For INT8, the length of the vector dimension must be a build-time constant.
        //!
        //! Deprecated usage:
        //!
        //! If running on the DLA, this format can be used for acceleration
        //! with the caveat that C must be less than or equal to 4.
        //! If used as DLA input and the build option kGPU_FALLBACK is not specified,
        //! it needs to meet line stride requirement of DLA format. Column stride in
        //! bytes must be a multiple of 64 on Orin.
        /// </remarks>
        kCHW4 = 3,

        /// <summary>
        /// 主向量格式，每个向量有16个标量
        /// Vector-major format with 16 scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第三个
        /// 此格式仅由DLA支持，需要FP16，并且至少有三个维度
        /// 此格式映射到FP16的本机特性格式，张量大小限制在C,H,W范围内[1,8192]
        /// The vector dimension is third to last.
        //!
        //! This format is only supported by DLA and requires FP16 and at least three dimensions.
        //! This format maps to the native feature format for FP16,
        //! and the tensor sizes are limited to C,H,W in the range [1,8192].
        /// </remarks>
        kCHW16 = 4,

        /// <summary>
        /// 主向量格式，每个向量有32个标量
        /// Vector-major format with 32 scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第三个
        /// 此格式需要INT8、FP32或FP16，并且至少有三个维度
        /// 对于DLA使用，此格式映射到INT8的本机特性格式，张量大小限制在C,H,W范围内[1,8192]
        /// The vector dimension is third to last.
        //!
        //! This format requires INT8, FP32, or FP16 and at least three dimensions.
        //!
        //! For DLA usage, this format maps to the native feature format for INT8,
        //! and the tensor sizes are limited to C,H,W in the range [1,8192].
        /// </remarks>
        kCHW32 = 5,

        /// <summary>
        /// 从向量格式，每个向量有八个标量
        /// Vector-minor format with eight scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第四个
        /// 此格式需要FP16或BF16，并且至少有四个维度
        /// The vector dimension is fourth to last.
        //!
        //! This format requires FP16 or BF16 and at least four dimensions.
        /// </remarks>
        kDHWC8 = 6,

        /// <summary>
        /// 主向量格式，每个向量有32个标量
        /// Vector-major format with 32 scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第四个
        /// 此格式需要FP16或INT8，并且至少有四个维度
        /// The vector dimension is fourth to last.
        //!
        //! This format requires FP16 or INT8 and at least four dimensions.
        /// </remarks>
        kCDHW32 = 7,

        /// <summary>
        /// 从向量格式，其中通道维度是倒数第三个且不填充
        /// Vector-minor format where channel dimension is third to last and unpadded
        /// </summary>
        /// <remarks>
        /// 此格式需要FP32或UINT8，并且至少有三个维度
        /// This format requires either FP32 or UINT8 and at least three dimensions.
        /// </remarks>
        kHWC = 8,

        /// <summary>
        /// DLA平面格式
        /// DLA planar format
        /// </summary>
        /// <remarks>
        /// 对于具有维度{N, C, H, W}的张量，W轴始终具有单位步幅
        /// 沿H轴步进的步幅四舍五入为64字节
        /// 内存布局等价于具有维度[N][C][H][roundUp(W, 64/elementSize)]的C数组
        /// 其中elementSize对于FP16为2，对于Int8为1，张量坐标(n, c, h, w)映射到数组下标[n][c][h][w]
        /// For a tensor with dimension {N, C, H, W}, the W axis
        //! always has unit stride. The stride for stepping along the H axis is
        //! rounded up to 64 bytes.
        //!
        //! The memory layout is equivalent to a C array with dimensions
        //! [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
        //! 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
        //! mapping to array subscript [n][c][h][w].
        /// </remarks>
        kDLA_LINEAR = 9,

        /// <summary>
        /// DLA图像格式
        /// DLA image format
        /// </summary>
        /// <remarks>
        /// 对于具有维度{N, C, H, W}的张量，C轴始终具有单位步幅
        /// 沿H轴步进的步幅在Orin上四舍五入为64字节
        /// C只能是1、3或4，如果C==1，则映射为灰度格式
        /// 如果C==3或C==4，则映射为彩色图像格式，如果C==3，沿W轴步进的步幅需要填充为4个元素
        /// 当C为{1,3,4}时，C'分别为{1,4,4}，内存布局等价于具有维度[N][H][roundUp(W, 64/C'/elementSize)][C']的C数组
        /// 其中elementSize对于FP16为2，对于Int8为1，张量坐标(n, c, h, w)映射到数组下标[n][h][w][c]
        /// For a tensor with dimension {N, C, H, W} the C axis
        //! always has unit stride. The stride for stepping along the H axis is rounded up
        //! to 64 bytes on Orin. C can only be 1, 3 or 4.
        //! If C == 1, it will map to grayscale format.
        //! If C == 3 or C == 4, it will map to color image format. And if C == 3,
        //! the stride for stepping along the W axis needs to be padded to 4 in elements.
        //!
        //! When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
        //! the memory layout is equivalent to a C array with dimensions
        //! [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
        //! where elementSize is 2 for FP16
        //! and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
        //! subscript [n][h][w][c].
        /// </remarks>
        kDLA_HWC4 = 10,

        /// <summary>
        /// 从向量格式，每个向量有16个标量
        /// Vector-minor format with 16 scalars per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第三个
        /// 此格式需要FP16、INT8或FP8，并且至少有三个维度
        /// The vector dimension is third to last.
        //!
        //! This requires FP16, INT8 or FP8 and at least three dimensions.
        /// </remarks>
        kHWC16 = 11,

        /// <summary>
        /// 从向量格式，每个向量有一个标量
        /// Vector-minor format with one scalar per vector
        /// </summary>
        /// <remarks>
        /// 向量维度是倒数第四个
        /// 此格式需要FP32，并且至少有四个维度
        /// The vector dimension is fourth to last.
        //!
        //! This format requires FP32 and at least four dimensions.
        /// </remarks>
        kDHWC = 12
    };

}
