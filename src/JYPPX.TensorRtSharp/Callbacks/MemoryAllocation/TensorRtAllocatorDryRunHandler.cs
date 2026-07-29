using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Handles allocator dry-run diagnostics.
/// 处理 allocator dry-run 诊断。
/// </summary>
/// <param name="request">The copied dry-run request. 复制后的 dry-run 请求。</param>
/// <returns>A copied dry-run result that never contains a device pointer. 不包含 device pointer 的 dry-run 结果副本。</returns>
public delegate TensorRtAllocatorDryRunResult TensorRtAllocatorDryRunHandler(TensorRtAllocatorDryRunRequest request);
