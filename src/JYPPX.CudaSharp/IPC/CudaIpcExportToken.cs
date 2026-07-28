using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies the CUDA resource represented by a copied IPC export token.
/// 标识复制型 CUDA IPC 导出 token 对应的资源类型。
/// </summary>
public enum CudaIpcExportTokenKind
{
    /// <summary>A CUDA event export token. CUDA event 导出 token。</summary>
    Event = 1,

    /// <summary>A CUDA device-memory export token. CUDA device memory 导出 token。</summary>
    Memory = 2
}

/// <summary>
/// Immutable managed copy of an opaque CUDA IPC export token.
/// CUDA IPC opaque 导出 token 的不可变托管副本。
/// </summary>
/// <remarks>
/// This value does not own the exported CUDA resource. Keep the source event or memory allocation alive
/// while another process uses the token. This API intentionally does not open imported resources.
/// 此值不拥有被导出的 CUDA 资源。其他进程使用 token 期间必须保持源 event 或 memory allocation 存活；
/// 本 API 有意不负责打开 imported resource。
/// </remarks>
public sealed class CudaIpcExportToken
{
    private const int TokenSizeInBytes = 64;
    private readonly byte[] _bytes;

    internal CudaIpcExportToken(CudaIpcExportTokenKind kind, byte[] bytes)
    {
        if (bytes == null)
        {
            throw new ArgumentNullException(nameof(bytes));
        }

        if (bytes.Length != TokenSizeInBytes)
        {
            throw new ArgumentException("CUDA IPC export token must contain exactly 64 bytes.", nameof(bytes));
        }

        if (!Enum.IsDefined(typeof(CudaIpcExportTokenKind), kind))
        {
            throw new ArgumentOutOfRangeException(nameof(kind));
        }

        Kind = kind;
        _bytes = (byte[])bytes.Clone();
    }

    /// <summary>Gets the exported CUDA resource kind. 获取导出的 CUDA 资源类型。</summary>
    public CudaIpcExportTokenKind Kind { get; }

    /// <summary>Gets the copied token length in bytes. 获取复制 token 的字节长度。</summary>
    public int Length => _bytes.Length;

    /// <summary>Creates an immutable token from transported bytes. 从传输后的字节创建不可变 token。</summary>
    /// <param name="kind">The CUDA resource kind represented by the token. token 表示的 CUDA 资源类型。</param>
    /// <param name="bytes">The opaque token bytes received from the exporting process. 从导出进程接收的 opaque token 字节。</param>
    /// <returns>An immutable managed token copy. 不可变的托管 token 副本。</returns>
    public static CudaIpcExportToken FromBytes(CudaIpcExportTokenKind kind, byte[] bytes) =>
        new CudaIpcExportToken(kind, bytes);

    /// <summary>Returns a new copy of the opaque token bytes. 返回 opaque token 字节的新副本。</summary>
    public byte[] ToArray() => (byte[])_bytes.Clone();

    /// <summary>Returns the opaque token as uppercase hexadecimal text. 将 opaque token 返回为大写十六进制文本。</summary>
    public string ToHexString()
    {
        const string alphabet = "0123456789ABCDEF";
        char[] characters = new char[_bytes.Length * 2];
        for (int index = 0; index < _bytes.Length; index++)
        {
            byte value = _bytes[index];
            characters[index * 2] = alphabet[value >> 4];
            characters[index * 2 + 1] = alphabet[value & 0x0F];
        }

        return new string(characters);
    }

    /// <summary>Returns a non-secret diagnostic summary. 返回不包含 token 内容的诊断摘要。</summary>
    public override string ToString() => $"Kind={Kind} Length={Length}";
}

/// <summary>
/// Immutable transport descriptor for an exported CUDA IPC memory allocation.
/// 导出 CUDA IPC memory allocation 的不可变传输 descriptor。
/// </summary>
/// <remarks>
/// The size must describe the original base allocation exactly. Treat the descriptor as one authenticated
/// transport unit and keep the exporting allocation alive until every importer has disposed its mapping.
/// size 必须精确描述原始 base allocation。应将 descriptor 作为一个整体进行可信传输，并保持导出
/// allocation 存活，直到所有 importer 都已释放 mapping。
/// </remarks>
public sealed class CudaIpcMemoryExportDescriptor
{
    internal CudaIpcMemoryExportDescriptor(CudaIpcExportToken token, int sizeInBytes)
    {
        if (token == null)
        {
            throw new ArgumentNullException(nameof(token));
        }

        if (token.Kind != CudaIpcExportTokenKind.Memory)
        {
            throw new ArgumentException("CUDA IPC memory descriptor requires a memory token.", nameof(token));
        }

        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        Token = token;
        SizeInBytes = sizeInBytes;
    }

    /// <summary>Gets the copied opaque memory token. 获取复制型 opaque memory token。</summary>
    public CudaIpcExportToken Token { get; }

    /// <summary>Gets the exported base-allocation size. 获取导出的 base allocation 大小。</summary>
    public int SizeInBytes { get; }

    /// <summary>Creates a descriptor from transported bytes and allocation metadata. 从传输字节和 allocation 元数据创建 descriptor。</summary>
    /// <param name="tokenBytes">The opaque memory-token bytes. opaque memory token 字节。</param>
    /// <param name="sizeInBytes">The exact exported allocation size. 导出 allocation 的精确大小。</param>
    /// <returns>An immutable transport descriptor. 不可变传输 descriptor。</returns>
    public static CudaIpcMemoryExportDescriptor FromBytes(byte[] tokenBytes, int sizeInBytes) =>
        new CudaIpcMemoryExportDescriptor(
            CudaIpcExportToken.FromBytes(CudaIpcExportTokenKind.Memory, tokenBytes),
            sizeInBytes);

    /// <summary>Returns a diagnostic summary without token contents. 返回不包含 token 内容的诊断摘要。</summary>
    public override string ToString() => $"Kind={Token.Kind} TokenLength={Token.Length} Size={SizeInBytes}";
}
