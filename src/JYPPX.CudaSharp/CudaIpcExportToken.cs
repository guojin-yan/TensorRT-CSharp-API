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
    private readonly byte[] _bytes;

    internal CudaIpcExportToken(CudaIpcExportTokenKind kind, byte[] bytes)
    {
        if (bytes == null)
        {
            throw new ArgumentNullException(nameof(bytes));
        }

        if (bytes.Length == 0)
        {
            throw new ArgumentException("CUDA IPC export token cannot be empty.", nameof(bytes));
        }

        Kind = kind;
        _bytes = (byte[])bytes.Clone();
    }

    /// <summary>Gets the exported CUDA resource kind. 获取导出的 CUDA 资源类型。</summary>
    public CudaIpcExportTokenKind Kind { get; }

    /// <summary>Gets the copied token length in bytes. 获取复制 token 的字节长度。</summary>
    public int Length => _bytes.Length;

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
