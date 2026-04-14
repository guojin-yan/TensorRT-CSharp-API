using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// Reduce层 - 对输入张量执行归约操作
    /// Reduce layer - performs reduction operation on the input tensor
    /// </summary>
    public class ReduceLayer : Layer
    {
        internal ReduceLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置归约操作类型
        /// Gets or sets the reduce operation type
        /// </summary>
        public TrtReduceOperation Operation
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtReduceLayer_getOperation(ptr, out TrtReduceOperation op));
                return op;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtReduceLayer_setOperation(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置要执行归约操作的轴位掩码
        /// Gets or sets the axis mask for the reduce computation
        /// </summary>
        public uint Axes
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtReduceLayer_getAxes(ptr, out uint axes));
                return axes;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtReduceLayer_setAxes(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置是否保留归约后的维度（保留为1）
        /// Gets or sets whether to keep the reduced dimensions (with size 1)
        /// </summary>
        public bool KeepDimensions
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtReduceLayer_getKeepDimensions(ptr, out bool keep));
                return keep;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtReduceLayer_setKeepDimensions(ptr, value));
            }
        }
    }
}
