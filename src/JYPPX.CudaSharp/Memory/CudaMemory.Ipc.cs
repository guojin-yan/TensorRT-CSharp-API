using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Copies an opaque CUDA IPC export token for this synchronous device allocation.
    /// 复制当前同步 device allocation 的 opaque CUDA IPC 导出 token。
    /// </summary>
    /// <remarks>
    /// Only allocations created by the public <see cref="CudaMemory(int)"/> constructor are exportable.
    /// Managed, asynchronous, and pool allocations are rejected. Keep this allocation alive while another
    /// process uses the token. This method never exposes the device pointer.
    /// 只有通过公开构造函数创建的同步分配可导出；managed、async 与 pool allocation 会被拒绝。
    /// 其他进程使用 token 期间必须保持当前分配存活；此方法绝不公开 device pointer。
    /// </remarks>
    public CudaIpcExportToken ExportIpcToken()
    {
        return NativeCudaApi.ExportMemoryIpcToken(_handle);
    }

    /// <summary>Creates a transport descriptor for this synchronous allocation. 为当前同步 allocation 创建传输 descriptor。</summary>
    /// <returns>The copied token and exact allocation size. 复制型 token 与精确 allocation 大小。</returns>
    public CudaIpcMemoryExportDescriptor ExportIpcDescriptor()
    {
        return new CudaIpcMemoryExportDescriptor(ExportIpcToken(), SizeInBytes);
    }

    /// <summary>Opens a process-local mapping from an exported CUDA IPC memory descriptor. 从导出的 CUDA IPC memory descriptor 打开进程内 mapping。</summary>
    /// <param name="descriptor">The token and exact allocation size transported from another process. 从其他进程传输的 token 与精确 allocation 大小。</param>
    /// <returns>An owner wrapper that closes the mapping with <c>cudaIpcCloseMemHandle</c>. 使用 <c>cudaIpcCloseMemHandle</c> 关闭 mapping 的 owner wrapper。</returns>
    public static CudaMemory ImportIpcDescriptor(CudaIpcMemoryExportDescriptor descriptor)
    {
        if (descriptor == null)
        {
            throw new ArgumentNullException(nameof(descriptor));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.ImportMemoryIpcDescriptor(descriptor), isIpcImported: true);
    }

    /// <summary>Tries to import a CUDA IPC memory descriptor and returns a diagnostic on CUDA failure. 尝试导入 CUDA IPC memory descriptor，并在 CUDA 失败时返回诊断。</summary>
    public static bool TryImportIpcDescriptor(
        CudaIpcMemoryExportDescriptor descriptor,
        out CudaMemory? memory,
        out string diagnostic)
    {
        try
        {
            memory = ImportIpcDescriptor(descriptor);
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            memory = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>Tries to copy an IPC export token and returns a diagnostic on failure. 尝试复制 IPC 导出 token，失败时返回诊断。</summary>
    public bool TryExportIpcToken(out CudaIpcExportToken? token, out string diagnostic)
    {
        try
        {
            token = ExportIpcToken();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            token = null;
            diagnostic = exception.Message;
            return false;
        }
    }

}
