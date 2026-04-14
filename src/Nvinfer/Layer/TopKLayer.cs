using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// TopK层 - 对输入应用TopK操作，返回K个最大值或最小值及其索引
    /// TopK layer - applies a TopK operation to its input, returning the K largest or smallest elements and their indices
    /// </summary>
    public class TopKLayer : Layer
    {
        internal TopKLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置TopK操作类型（最大值或最小值）
        /// Gets or sets the TopK operation type (maximum or minimum)
        /// </summary>
        public TrtTopKOperation Operation
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtTopKLayer_getOperation(ptr, out TrtTopKOperation op));
                return op;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtTopKLayer_setOperation(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置K值（要返回的元素数量）
        /// Gets or sets the K value (number of elements to return)
        /// </summary>
        public int K
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtTopKLayer_getK(ptr, out int k));
                return k;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtTopKLayer_setK(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置进行TopK操作的轴
        /// Gets or sets the axes on which to perform the TopK operation
        /// </summary>
        public uint Axes
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtTopKLayer_getAxes(ptr, out uint axes));
                return axes;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtTopKLayer_setAxes(ptr, value));
            }
        }
    }
}
