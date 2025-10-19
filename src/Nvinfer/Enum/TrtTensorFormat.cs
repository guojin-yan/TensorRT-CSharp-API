using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    public enum TrtTensorFormat : int
    {
        //! Memory layout is similar to an array in C or C++.
        //! The stride of each dimension is the product of the dimensions after it.
        //! The last dimension has unit stride.
        //!
        //! This format supports all TensorRT types.
        //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
        kLINEAR = 0,

        //! Vector-major format with two scalars per vector.
        //! Vector dimension is third to last.
        //!
        //! This format requires FP16 or BF16 and at least three dimensions.
        kCHW2 = 1,

        //! Vector-minor format with eight scalars per vector.
        //! Vector dimension is third to last.
        //! This format requires FP16 or BF16 and at least three dimensions.
        kHWC8 = 2,

        //! Vector-major format with four scalars per vector.
        //! Vector dimension is third to last.
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
        kCHW4 = 3,

        //! Vector-major format with 16 scalars per vector.
        //! Vector dimension is third to last.
        //!
        //! This format is only supported by DLA and requires FP16 and at least three dimensions.
        //! This format maps to the native feature format for FP16,
        //! and the tensor sizes are limited to C,H,W in the range [1,8192].
        kCHW16 = 4,

        //! Vector-major format with 32 scalars per vector.
        //! Vector dimension is third to last.
        //!
        //! This format requires INT8, FP32, or FP16 and at least three dimensions.
        //!
        //! For DLA usage, this format maps to the native feature format for INT8,
        //! and the tensor sizes are limited to C,H,W in the range [1,8192].
        kCHW32 = 5,

        //! Vector-minor format with eight scalars per vector.
        //! Vector dimension is fourth to last.
        //!
        //! This format requires FP16 or BF16 and at least four dimensions.
        kDHWC8 = 6,

        //! Vector-major format with 32 scalars per vector.
        //! Vector dimension is fourth to last.
        //!
        //! This format requires FP16 or INT8 and at least four dimensions.
        kCDHW32 = 7,

        //! Vector-minor format where channel dimension is third to last and unpadded.
        //!
        //! This format requires either FP32 or UINT8 and at least three dimensions.
        kHWC = 8,

        //! DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
        //! always has unit stride. The stride for stepping along the H axis is
        //! rounded up to 64 bytes.
        //!
        //! The memory layout is equivalent to a C array with dimensions
        //! [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
        //! 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
        //! mapping to array subscript [n][c][h][w].
        kDLA_LINEAR = 9,

        //! DLA image format. For a tensor with dimension {N, C, H, W} the C axis
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
        kDLA_HWC4 = 10,

        //! Vector-minor format with 16 scalars per vector.
        //! Vector dimension is third to last.
        //!
        //! This requires FP16, INT8 or FP8 and at least three dimensions.
        kHWC16 = 11,

        //! Vector-minor format with one scalar per vector.
        //! Vector dimension is fourth to last.
        //!
        //! This format requires FP32 and at least four dimensions.
        kDHWC = 12
    };
}
