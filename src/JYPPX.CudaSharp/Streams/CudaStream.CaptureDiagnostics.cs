using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaStream
{
    /// <summary>
    /// Gets CUDA graph-capture metadata for this stream.
    /// 获取当前 stream 的 CUDA graph 捕获元数据。
    /// </summary>
    /// <returns>The current capture status and capture id. 当前捕获状态和捕获 ID。</returns>
    public CudaStreamCaptureInfo GetCaptureInfo()
    {
        return NativeCudaApi.GetStreamCaptureInfo(_handle);
    }

    /// <summary>
    /// Tries to read CUDA graph-capture metadata without throwing when the runtime or symbol is unavailable.
    /// 在 CUDA runtime 或相关符号不可用时以诊断字符串返回失败，而不是抛出异常。
    /// </summary>
    /// <param name="captureInfo">The capture metadata when the query succeeds. 查询成功时返回的 capture 元数据。</param>
    /// <param name="diagnostic">An empty string on success, or the CUDA bridge diagnostic on failure. 成功时为空字符串，失败时为 CUDA 桥接诊断。</param>
    /// <returns><c>true</c> when capture metadata was read successfully. 成功读取 capture 元数据时返回 <c>true</c>。</returns>
    public bool TryGetCaptureInfo(out CudaStreamCaptureInfo captureInfo, out string diagnostic)
    {
        try
        {
            captureInfo = GetCaptureInfo();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            captureInfo = default;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets the scalar capture metadata from CUDA's per-thread stream variant.
    /// 获取 CUDA per-thread stream 变体返回的标量 capture 元数据。
    /// </summary>
    /// <remarks>
    /// The vendor ptsz entry point reports only status and id; graph and dependency outputs are not part of this API.
    /// vendor ptsz 入口只报告 status 和 id；graph 与 dependency 输出不属于该 API。
    /// </remarks>
    public CudaStreamCaptureScalarInfo GetCaptureInfoPtzs()
    {
        return NativeCudaApi.GetStreamCaptureInfoPtzs(_handle);
    }

    /// <summary>
    /// Gets a pointer-free CUDA 13 resource snapshot associated with this stream.
    /// 获取与此 stream 关联的 CUDA 13 资源无指针快照。
    /// </summary>
    public CudaDevResourceSnapshot GetDevResourceSnapshot(CudaDevResourceType resourceType)
    {
        return NativeCudaApi.GetStreamDevResourceSnapshot(_handle, resourceType);
    }

    /// <summary>
    /// Tries to read scalar capture metadata from CUDA's per-thread stream variant.
    /// 尝试读取 CUDA per-thread stream 变体返回的标量 capture 元数据。
    /// </summary>
    public bool TryGetCaptureInfoPtzs(out CudaStreamCaptureScalarInfo captureInfo, out string diagnostic)
    {
        try
        {
            captureInfo = GetCaptureInfoPtzs();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            captureInfo = default;
            diagnostic = exception.Message;
            return false;
        }
    }

}
