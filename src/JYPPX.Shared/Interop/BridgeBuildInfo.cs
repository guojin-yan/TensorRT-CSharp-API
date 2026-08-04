namespace JYPPX.TensorRtSharp.Shared.Interop;

/// <summary>
/// Build-time metadata reported by the native bridge.
/// 原生 bridge 报告的构建期元数据。
/// </summary>
public sealed class BridgeBuildInfo
{
    /// <summary>
    /// Creates a bridge build-information snapshot.
    /// 创建 bridge 构建信息快照。
    /// </summary>
    public BridgeBuildInfo(
        int abiVersion,
        int bridgeVersionMajor,
        int bridgeVersionMinor,
        int bridgeVersionPatch,
        string bridgeName,
        string bridgeBanner,
        string compilerId,
        string compilerVersion,
        string systemName,
        string systemProcessor,
        string buildConfiguration,
        string cudaToolkitVersion,
        string tensorRtVersion,
        bool hasCudaToolkit,
        bool hasTensorRt,
        bool cudaBindingsEnabled,
        bool tensorRtBindingsEnabled)
    {
        AbiVersion = abiVersion;
        BridgeVersionMajor = bridgeVersionMajor;
        BridgeVersionMinor = bridgeVersionMinor;
        BridgeVersionPatch = bridgeVersionPatch;
        BridgeName = bridgeName;
        BridgeBanner = bridgeBanner;
        CompilerId = compilerId;
        CompilerVersion = compilerVersion;
        SystemName = systemName;
        SystemProcessor = systemProcessor;
        BuildConfiguration = buildConfiguration;
        CudaToolkitVersion = cudaToolkitVersion;
        TensorRtVersion = tensorRtVersion;
        HasCudaToolkit = hasCudaToolkit;
        HasTensorRt = hasTensorRt;
        CudaBindingsEnabled = cudaBindingsEnabled;
        TensorRtBindingsEnabled = tensorRtBindingsEnabled;
    }

    /// <summary>
    /// Gets the native bridge ABI version.
    /// 获取原生 bridge ABI 版本。
    /// </summary>
    public int AbiVersion { get; }
    /// <summary>
    /// Gets the bridge semantic-version major component.
    /// 获取 bridge 语义版本的主版本号。
    /// </summary>
    public int BridgeVersionMajor { get; }
    /// <summary>
    /// Gets the bridge semantic-version minor component.
    /// 获取 bridge 语义版本的次版本号。
    /// </summary>
    public int BridgeVersionMinor { get; }
    /// <summary>
    /// Gets the bridge semantic-version patch component.
    /// 获取 bridge 语义版本的补丁号。
    /// </summary>
    public int BridgeVersionPatch { get; }
    /// <summary>
    /// Gets the logical bridge name.
    /// 获取逻辑 bridge 名称。
    /// </summary>
    public string BridgeName { get; }
    /// <summary>
    /// Gets the bridge banner string.
    /// 获取 bridge banner 字符串。
    /// </summary>
    public string BridgeBanner { get; }
    /// <summary>
    /// Gets the compiler identifier used to build the bridge.
    /// 获取构建 bridge 时使用的编译器标识。
    /// </summary>
    public string CompilerId { get; }
    /// <summary>
    /// Gets the compiler version used to build the bridge.
    /// 获取构建 bridge 时使用的编译器版本。
    /// </summary>
    public string CompilerVersion { get; }
    /// <summary>
    /// Gets the operating system name used for the bridge build.
    /// 获取构建 bridge 时使用的操作系统名称。
    /// </summary>
    public string SystemName { get; }
    /// <summary>
    /// Gets the target processor string used for the bridge build.
    /// 获取构建 bridge 时使用的目标处理器字符串。
    /// </summary>
    public string SystemProcessor { get; }
    /// <summary>
    /// Gets the build configuration, such as Debug or Release.
    /// 获取构建配置，例如 Debug 或 Release。
    /// </summary>
    public string BuildConfiguration { get; }
    /// <summary>
    /// Gets the CUDA toolkit version used for the bridge build.
    /// 获取构建 bridge 时使用的 CUDA toolkit 版本。
    /// </summary>
    public string CudaToolkitVersion { get; }
    /// <summary>
    /// Gets the TensorRT version used for the bridge build.
    /// 获取构建 bridge 时使用的 TensorRT 版本。
    /// </summary>
    public string TensorRtVersion { get; }
    /// <summary>
    /// Gets whether CUDA toolkit support was enabled for the bridge build.
    /// 获取 bridge 构建时是否启用了 CUDA toolkit 支持。
    /// </summary>
    public bool HasCudaToolkit { get; }
    /// <summary>
    /// Gets whether TensorRT support was enabled for the bridge build.
    /// 获取 bridge 构建时是否启用了 TensorRT 支持。
    /// </summary>
    public bool HasTensorRt { get; }
    /// <summary>
    /// Gets whether CUDA managed bindings were emitted.
    /// 获取是否生成了 CUDA 托管绑定。
    /// </summary>
    public bool CudaBindingsEnabled { get; }
    /// <summary>
    /// Gets whether TensorRT managed bindings were emitted.
    /// 获取是否生成了 TensorRT 托管绑定。
    /// </summary>
    public bool TensorRtBindingsEnabled { get; }
}
