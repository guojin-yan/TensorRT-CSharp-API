using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Possible modes for stream capture thread interactions. For more details see
    /// ::cudaStreamBeginCapture and ::cudaThreadExchangeStreamCaptureMode
    /// 流捕获线程交互的可能模式。有关更多详细信息，请参阅
    /// ::cudaStreamBeginCapture 和 ::cudaThreadExchangeStreamCaptureMode
    /// </summary>
    public enum CudaStreamCaptureMode
    {
        /// <summary>
        /// Default mode (Global)
        /// 默认模式（全局）
        /// </summary>
        Global = 0,
        /// <summary>
        /// Thread local mode
        /// 线程局部模式
        /// </summary>
        ThreadLocal = 1,
        /// <summary>
        /// Relaxed mode
        /// 宽松模式
        /// </summary>
        Relaxed = 2
    }
}
