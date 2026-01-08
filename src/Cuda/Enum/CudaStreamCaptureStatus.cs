using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Possible stream capture statuses returned by ::cudaStreamIsCapturing
    /// ::cudaStreamIsCapturing 返回的可能流捕获状态
    /// </summary>
    public enum CudaStreamCaptureStatus
    {
        /// <summary>
        /// Stream is not capturing
        /// 流未处于捕获状态
        /// </summary>
        None = 0,
        /// <summary>
        /// Stream is actively capturing
        /// 流正在主动捕获
        /// </summary>
        Active = 1,
        /// <summary>
        /// Stream is part of a capture sequence that has been invalidated, but not terminated
        /// 流是已失效但尚未终止的捕获序列的一部分
        /// </summary>
        Invalidated = 2
    }
}
