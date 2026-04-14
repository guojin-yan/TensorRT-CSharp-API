using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 洗牌层 - 对输入张量进行转置、重塑和第二次转置操作
    /// Shuffle layer - performs transpose, reshape, and second transpose operations on the input tensor
    /// </summary>
    public class ShuffleLayer : Layer
    {
        internal ShuffleLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置第一次转置的维度排列
        /// Gets or sets the permutation for the first transpose operation
        /// </summary>
        public Dims FirstTranspose
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_getFirstTranspose(ptr, out Dims permutation));
                return permutation;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_setFirstTranspose(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置重塑后的维度
        /// Gets or sets the dimensions for the reshape operation
        /// </summary>
        public Dims ReshapeDimensions
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_getReshapeDimensions(ptr, out Dims dimensions));
                return dimensions;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_setReshapeDimensions(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置第二次转置的维度排列
        /// Gets or sets the permutation for the second transpose operation
        /// </summary>
        public Dims SecondTranspose
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_getSecondTranspose(ptr, out Dims permutation));
                return permutation;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_setSecondTranspose(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置零是否作为占位符
        /// 当设置为true时，重塑维度中的0表示继承输入张量对应维度的值
        /// Gets or sets whether zero is a placeholder
        /// When set to true, a zero in reshape dimensions means inherit the value from the corresponding dimension of the input tensor
        /// </summary>
        public bool ZeroIsPlaceholder
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_getZeroIsPlaceholder(ptr, out int zeroIsPlaceholder));
                return zeroIsPlaceholder != 0;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtShuffleLayer_setZeroIsPlaceholder(ptr, value ? 1 : 0));
            }
        }
    }
}
