using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Text;

namespace JYPPX.CudaSharp;

/// <summary>Stable NVRTC result codes returned as diagnostic data. 作为诊断数据返回的稳定 NVRTC 结果码。</summary>
public enum CudaRtcResultCode
{
    /// <summary>Compilation succeeded. 编译成功。</summary>
    Success = 0,
    /// <summary>NVRTC could not allocate memory. NVRTC 无法分配内存。</summary>
    OutOfMemory = 1,
    /// <summary>Program creation failed. program 创建失败。</summary>
    ProgramCreationFailure = 2,
    /// <summary>An input was invalid. 输入无效。</summary>
    InvalidInput = 3,
    /// <summary>The program state was invalid. program 状态无效。</summary>
    InvalidProgram = 4,
    /// <summary>A compile option was invalid. compile option 无效。</summary>
    InvalidOption = 5,
    /// <summary>CUDA C++ compilation failed; inspect the log. CUDA C++ 编译失败；应检查日志。</summary>
    Compilation = 6,
    /// <summary>A built-in operation failed. 内建操作失败。</summary>
    BuiltinOperationFailure = 7,
    /// <summary>Name expressions were added after compilation. name expression 添加时机无效。</summary>
    NoNameExpressionsAfterCompilation = 8,
    /// <summary>Lowered names were queried before compilation. lowered name 查询时机无效。</summary>
    NoLoweredNamesBeforeCompilation = 9,
    /// <summary>A name expression was invalid. name expression 无效。</summary>
    NameExpressionNotValid = 10,
    /// <summary>NVRTC reported an internal error. NVRTC 报告内部错误。</summary>
    InternalError = 11,
    /// <summary>NVRTC could not write a time file. NVRTC 无法写入 time file。</summary>
    TimeFileWriteFailed = 12,
    /// <summary>No PCH create attempt occurred. 未尝试创建 PCH。</summary>
    NoPchCreateAttempted = 13,
    /// <summary>The PCH heap was exhausted. PCH heap 已耗尽。</summary>
    PchCreateHeapExhausted = 14,
    /// <summary>PCH creation failed. PCH 创建失败。</summary>
    PchCreate = 15,
    /// <summary>Compilation was cancelled. 编译被取消。</summary>
    Cancelled = 16,
    /// <summary>NVRTC could not write a time-trace file. NVRTC 无法写入 time trace file。</summary>
    TimeTraceFileWriteFailed = 17
}

/// <summary>Identifies a copied runtime-compilation artifact. 标识复制型 runtime compilation 工件。</summary>
public enum CudaRtcArtifactKind
{
    /// <summary>Null-terminated PTX bytes exactly as returned by NVRTC. NVRTC 原样返回的以 NUL 结尾的 PTX 字节。</summary>
    Ptx = 1,
    /// <summary>CUBIN bytes for a real SM target. 针对真实 SM target 的 CUBIN 字节。</summary>
    Cubin = 2,
    /// <summary>LTO IR bytes when link-time optimization output is enabled. 启用 link-time optimization 输出时的 LTO IR 字节。</summary>
    LtoIr = 3
}

/// <summary>Copied immutable NVRTC artifact with reproducibility metadata. 带可复现元数据的不可变 NVRTC 工件副本。</summary>
public sealed class CudaRtcArtifact
{
    private readonly byte[] _bytes;

    internal CudaRtcArtifact(
        CudaRtcArtifactKind kind,
        byte[] bytes,
        string sourceSha256,
        string headersSha256,
        string optionsSha256,
        string compilerVersion,
        string targetArchitecture)
    {
        Kind = kind;
        _bytes = (byte[])bytes.Clone();
        Sha256 = CudaRtcHash.Bytes(_bytes);
        SourceSha256 = sourceSha256;
        HeadersSha256 = headersSha256;
        OptionsSha256 = optionsSha256;
        CompilerVersion = compilerVersion;
        TargetArchitecture = targetArchitecture;
    }

    /// <summary>Gets the artifact kind. 获取工件类型。</summary>
    public CudaRtcArtifactKind Kind { get; }

    /// <summary>Gets the copied payload length. 获取复制 payload 长度。</summary>
    public int Length => _bytes.Length;

    /// <summary>Gets the artifact SHA256. 获取工件 SHA256。</summary>
    public string Sha256 { get; }

    /// <summary>Gets the source SHA256. 获取 source SHA256。</summary>
    public string SourceSha256 { get; }

    /// <summary>Gets the canonical virtual-header SHA256. 获取规范化虚拟 header SHA256。</summary>
    public string HeadersSha256 { get; }

    /// <summary>Gets the canonical compile-option SHA256. 获取规范化 compile option SHA256。</summary>
    public string OptionsSha256 { get; }

    /// <summary>Gets the loaded NVRTC compiler version. 获取已加载 NVRTC compiler version。</summary>
    public string CompilerVersion { get; }

    /// <summary>Gets the requested target architecture, or an empty string. 获取请求的 target architecture；未指定时为空。</summary>
    public string TargetArchitecture { get; }

    /// <summary>Returns a new payload copy. 返回新的 payload 副本。</summary>
    public byte[] ToArray() => (byte[])_bytes.Clone();
}

/// <summary>Copied mapping from a source name expression to its lowered name. source name expression 到 lowered name 的复制型映射。</summary>
public sealed class CudaRtcLoweredName
{
    internal CudaRtcLoweredName(string expression, string loweredName)
    {
        Expression = expression;
        LoweredName = loweredName;
    }

    /// <summary>Gets the original expression. 获取原始 expression。</summary>
    public string Expression { get; }

    /// <summary>Gets the copied lowered name. 获取复制的 lowered name。</summary>
    public string LoweredName { get; }
}

/// <summary>Immutable NVRTC compilation result retaining failure logs and copied success artifacts. 保留失败日志与成功工件副本的不可变 NVRTC 编译结果。</summary>
public sealed class CudaRtcCompilationResult
{
    private readonly ReadOnlyCollection<CudaRtcArtifact> _artifacts;
    private readonly ReadOnlyCollection<CudaRtcLoweredName> _loweredNames;

    internal CudaRtcCompilationResult(
        CudaRtcResultCode resultCode,
        string log,
        string compilerVersion,
        IEnumerable<CudaRtcArtifact> artifacts,
        IEnumerable<CudaRtcLoweredName> loweredNames)
    {
        ResultCode = resultCode;
        Log = log;
        CompilerVersion = compilerVersion;
        _artifacts = new ReadOnlyCollection<CudaRtcArtifact>(new List<CudaRtcArtifact>(artifacts));
        _loweredNames = new ReadOnlyCollection<CudaRtcLoweredName>(new List<CudaRtcLoweredName>(loweredNames));
    }

    /// <summary>Gets whether compilation completed successfully. 获取编译是否成功完成。</summary>
    public bool Success => ResultCode == CudaRtcResultCode.Success;

    /// <summary>Gets the NVRTC result code. 获取 NVRTC 结果码。</summary>
    public CudaRtcResultCode ResultCode { get; }

    /// <summary>Gets the complete copied compiler log. 获取完整复制的 compiler log。</summary>
    public string Log { get; }

    /// <summary>Gets the loaded compiler version. 获取已加载 compiler version。</summary>
    public string CompilerVersion { get; }

    /// <summary>Gets available artifacts; unavailable option-dependent outputs are omitted. 获取可用工件；依赖 option 且不可用的输出不会出现。</summary>
    public IReadOnlyList<CudaRtcArtifact> Artifacts => _artifacts;

    /// <summary>Gets copied lowered names in source-expression order. 获取按 source expression 顺序排列的 lowered name 副本。</summary>
    public IReadOnlyList<CudaRtcLoweredName> LoweredNames => _loweredNames;

    /// <summary>Finds an available artifact by kind. 按类型查找可用工件。</summary>
    public CudaRtcArtifact? FindArtifact(CudaRtcArtifactKind kind)
    {
        foreach (CudaRtcArtifact artifact in _artifacts)
        {
            if (artifact.Kind == kind)
            {
                return artifact;
            }
        }
        return null;
    }
}

internal static class CudaRtcHash
{
    public static string Text(string value) => Bytes(Encoding.UTF8.GetBytes(value));

    public static string Sequence(IEnumerable<string> values)
    {
        var canonical = new StringBuilder();
        foreach (string value in values)
        {
            canonical.Append(value.Length).Append(':').Append(value).Append(';');
        }
        return Text(canonical.ToString());
    }

    public static string Bytes(byte[] bytes)
    {
        byte[] hash;
        using (SHA256 algorithm = SHA256.Create())
        {
            hash = algorithm.ComputeHash(bytes);
        }
        const string alphabet = "0123456789abcdef";
        char[] result = new char[hash.Length * 2];
        for (int index = 0; index < hash.Length; index++)
        {
            result[index * 2] = alphabet[hash[index] >> 4];
            result[index * 2 + 1] = alphabet[hash[index] & 0x0F];
        }
        return new string(result);
    }
}
