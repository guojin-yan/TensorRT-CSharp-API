using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Copied optional-NVRTC capability snapshot. 可选 NVRTC 的复制型能力快照。</summary>
public sealed class CudaRtcCapability
{
    internal CudaRtcCapability(NativeCudaRtcCapabilityInfo info, string loadedLibraryName, string dependencyDiagnostic)
    {
        IsAvailable = info.DependencyAvailable != 0;
        VersionMajor = info.VersionMajor;
        VersionMinor = info.VersionMinor;
        SupportsPtx = info.SupportsPtx != 0;
        SupportsCubin = info.SupportsCubin != 0;
        SupportsLtoIr = info.SupportsLtoIr != 0;
        SupportsDeprecatedNvvm = info.SupportsDeprecatedNvvm != 0;
        SupportsNameExpressions = info.SupportsNameExpressions != 0;
        LoadedLibraryName = loadedLibraryName;
        DependencyDiagnostic = dependencyDiagnostic;
    }

    /// <summary>Gets whether a complete compatible NVRTC library was loaded. 获取是否已加载完整兼容的 NVRTC library。</summary>
    public bool IsAvailable { get; }
    /// <summary>Gets the compiler major version. 获取 compiler major version。</summary>
    public int VersionMajor { get; }
    /// <summary>Gets the compiler minor version. 获取 compiler minor version。</summary>
    public int VersionMinor { get; }
    /// <summary>Gets whether PTX output is supported. 获取是否支持 PTX 输出。</summary>
    public bool SupportsPtx { get; }
    /// <summary>Gets whether CUBIN output is supported. 获取是否支持 CUBIN 输出。</summary>
    public bool SupportsCubin { get; }
    /// <summary>Gets whether LTO IR output is supported. 获取是否支持 LTO IR 输出。</summary>
    public bool SupportsLtoIr { get; }
    /// <summary>Gets whether deprecated NVVM output symbols exist. 获取 deprecated NVVM 输出 symbol 是否存在。</summary>
    public bool SupportsDeprecatedNvvm { get; }
    /// <summary>Gets whether name-expression lowering is supported. 获取是否支持 name expression lowering。</summary>
    public bool SupportsNameExpressions { get; }
    /// <summary>Gets the actual loader candidate used, without exposing a module handle. 获取实际使用的 loader candidate，不暴露 module handle。</summary>
    public string LoadedLibraryName { get; }
    /// <summary>Gets an actionable diagnostic when unavailable. 获取不可用时的可操作诊断。</summary>
    public string DependencyDiagnostic { get; }
    /// <summary>Gets the dotted compiler version, or an empty string when unavailable. 获取点分 compiler version；不可用时为空。</summary>
    public string Version => IsAvailable ? VersionMajor + "." + VersionMinor : string.Empty;
}

/// <summary>High-level entry point for optional CUDA runtime compilation. 可选 CUDA runtime compilation 的高层入口。</summary>
public static class CudaRtcCompiler
{
    /// <summary>Queries NVRTC without making it a load-time dependency of the core bridge. 查询 NVRTC，但不把它变成 core bridge 的加载时依赖。</summary>
    public static CudaRtcCapability GetCapability()
    {
        NativeBridgeLoader.EnsureInitialized();
        return new CudaRtcCapability(
            NativeCudaApi.QueryRtcCapability(),
            NativeCudaApi.GetRtcLoadedLibraryName(),
            NativeCudaApi.GetRtcDependencyDiagnostic());
    }

    /// <summary>Compiles a source snapshot once while retaining failure logs as data. 一次性编译 source 快照，并把失败日志保留为数据。</summary>
    public static CudaRtcCompilationResult Compile(CudaRtcProgramSource source, CudaRtcCompileOptions? options = null)
    {
        using (var program = new CudaRtcProgram(source))
        {
            return program.Compile(options ?? CudaRtcCompileOptions.Default);
        }
    }
}
