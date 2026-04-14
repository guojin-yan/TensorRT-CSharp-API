using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 切片层 - 对输入张量进行切片操作
    /// Slice layer - performs slice operations on the input tensor
    /// </summary>
    public class SliceLayer : Layer
    {
        internal SliceLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置切片的起始位置
        /// Gets or sets the start coordinates for the slice
        /// </summary>
        public Dims Start
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_getStart(ptr, out Dims start));
                return start;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_setStart(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置切片的大小
        /// Gets or sets the size dimensions for the slice
        /// </summary>
        public Dims Size
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_getSize(ptr, out Dims size));
                return size;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_setSize(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置切片的步长
        /// Gets or sets the stride for the slice
        /// </summary>
        public Dims Stride
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_getStride(ptr, out Dims stride));
                return stride;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_setStride(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置切片模式
        /// Gets or sets the slice mode
        /// </summary>
        public TrtSliceMode Mode
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_getMode(ptr, out TrtSliceMode mode));
                return mode;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtSliceLayer_setMode(ptr, value));
            }
        }
    }
}
