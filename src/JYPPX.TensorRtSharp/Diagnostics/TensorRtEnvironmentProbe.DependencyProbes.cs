using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Queries the current bridge state without requiring TensorRT inference to succeed.
/// 在不要求 TensorRT 推理成功的前提下查询当前 bridge 状态。
/// </summary>
public static partial class TensorRtEnvironmentProbe
{
    /// <summary>
    /// Probes native bridge, TensorRT, CUDA, cuDNN, and parser DLL resolution using managed OS APIs only.
    /// 仅使用托管 OS API 探测 native bridge、TensorRT、CUDA、cuDNN 与 parser DLL 的解析情况。
    /// </summary>
    /// <remarks>
    /// This diagnostic does not call TensorRT global version functions, global registry functions, runtime creation, or builder creation.
    /// It is intended to inspect DLL search-path drift before riskier vendor entry points are used.
    /// 该诊断不会调用 TensorRT 全局版本函数、全局 registry 函数、runtime 创建或 builder 创建；用于在调用风险更高的 vendor 入口点前检查 DLL 搜索路径漂移。
    /// </remarks>
    /// <param name="line">The TensorRT API line used to select expected TensorRT DLL file names. 用于选择预期 TensorRT DLL 文件名的 TensorRT API line。</param>
    /// <returns>A non-throwing native dependency probe report. 非抛异常的 native 依赖探测报告。</returns>
    public static TensorRtDependencyProbeReport ProbeNativeDependencies(TensorRtApiLine line)
    {
        List<string> diagnostics = new List<string>();
        bool bridgeInitialized;
        string bridgeDiagnostic;

        try
        {
            NativeBridgeLoader.EnsureInitialized();
            bridgeInitialized = true;
            bridgeDiagnostic = "Bridge loader initialized.";
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            bridgeInitialized = false;
            bridgeDiagnostic = FormatProbeException("Bridge initialization", exception);
        }

        IReadOnlyList<TensorRtNativeDependencyInfo> bridgeCandidates = EnumerateNativeBridgeCandidates(diagnostics);
        IReadOnlyList<TensorRtNativeDependencyInfo> loadedModules = EnumerateLoadedDependencyModules(diagnostics);
        IReadOnlyList<TensorRtNativeDependencyInfo> searchPathCandidates = EnumerateSearchPathDependencyCandidates(line, diagnostics);

        return new TensorRtDependencyProbeReport(line, bridgeInitialized, bridgeDiagnostic, bridgeCandidates, loadedModules, searchPathCandidates, diagnostics);
    }

    /// <summary>
    /// Runs staged TensorRT runtime probes and returns diagnostics for each stage.
    /// 运行 TensorRT runtime 分阶段探针，并返回每个阶段的诊断信息。
    /// </summary>
    /// <remarks>
    /// The probe intentionally separates global version, global registry, logger creation, builder creation, and runtime creation.
    /// 探针会刻意分离全局版本、全局 registry、logger 创建、builder 创建和 runtime 创建阶段。
    /// </remarks>
    /// <param name="line">The TensorRT API line to probe. 要探测的 TensorRT API line。</param>
    /// <returns>A staged runtime probe report. 分阶段 runtime 探针报告。</returns>
    public static TensorRtRuntimeProbeReport ProbeRuntime(TensorRtApiLine line)
    {
        List<TensorRtRuntimeProbeStage> stages = new List<TensorRtRuntimeProbeStage>();
        TensorRtGlobalRuntimeVersion? version = null;
        TensorRtPluginRegistryInventory? globalRegistry = null;

        try
        {
            NativeBridgeLoader.EnsureInitialized();
            stages.Add(new TensorRtRuntimeProbeStage("BridgeInitialized", succeeded: true, "Bridge loader initialized."));
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            stages.Add(new TensorRtRuntimeProbeStage("BridgeInitialized", succeeded: false, FormatProbeException("Bridge initialization", exception)));
            return new TensorRtRuntimeProbeReport(line, version, globalRegistry, stages);
        }

        bool inferVersionOk = TryAddProbeStage(stages, "GlobalInferLibVersion", () => NativeBridgeApi.GetGlobalInferLibVersion(line), value => $"Packed={value}", out int inferLibVersion);
        bool inferMajorOk = TryAddProbeStage(stages, "GlobalInferLibMajorVersion", () => NativeBridgeApi.GetGlobalInferLibMajorVersion(line), value => $"Major={value}", out int inferMajor);
        bool inferMinorOk = TryAddProbeStage(stages, "GlobalInferLibMinorVersion", () => NativeBridgeApi.GetGlobalInferLibMinorVersion(line), value => $"Minor={value}", out int inferMinor);
        bool inferPatchOk = TryAddProbeStage(stages, "GlobalInferLibPatchVersion", () => NativeBridgeApi.GetGlobalInferLibPatchVersion(line), value => $"Patch={value}", out int inferPatch);
        bool inferBuildOk = TryAddProbeStage(stages, "GlobalInferLibBuildVersion", () => NativeBridgeApi.GetGlobalInferLibBuildVersion(line), value => $"Build={value}", out int inferBuild);
        bool onnxParserOk = TryAddProbeStage(stages, "GlobalOnnxParserVersion", () => NativeBridgeApi.GetGlobalOnnxParserVersion(line), value => $"OnnxParser={value}", out int onnxParserVersion);
        bool globalLoggerOk = TryAddProbeStage(stages, "GlobalLogger", () => NativeBridgeApi.GlobalHasLogger(line), value => $"HasGlobalLogger={value}", out bool hasGlobalLogger);
        if (inferVersionOk && inferMajorOk && inferMinorOk && inferPatchOk && inferBuildOk && onnxParserOk && globalLoggerOk)
        {
            version = new TensorRtGlobalRuntimeVersion(line, inferLibVersion, inferMajor, inferMinor, inferPatch, inferBuild, onnxParserVersion, hasGlobalLogger);
        }

        if (TryGetGlobalPluginRegistryInventory(line, includeCreatorFields: false, out globalRegistry, out string registryDiagnostic))
        {
            string recursiveCount = globalRegistry?.RecursiveCreatorCount?.ToString() ?? "n/a";
            stages.Add(new TensorRtRuntimeProbeStage("GlobalPluginRegistry", succeeded: true, $"Creators={globalRegistry?.CreatorCount ?? 0} Recursive={recursiveCount} ParentSearch={globalRegistry?.ParentSearchEnabled ?? false} ErrorRecorder={globalRegistry?.HasErrorRecorder ?? false}"));
        }
        else
        {
            stages.Add(new TensorRtRuntimeProbeStage("GlobalPluginRegistry", succeeded: false, registryDiagnostic));
        }

        bool loggerOk = TryCreateLogger(line, out string loggerMessage);
        stages.Add(new TensorRtRuntimeProbeStage("LoggerCreate", loggerOk, loggerMessage));

        bool builderOk = TryCreateBuilder(line, out string builderMessage);
        stages.Add(new TensorRtRuntimeProbeStage("BuilderCreate", builderOk, builderMessage));

        bool runtimeOk = TryCreateRuntime(line, out string runtimeMessage);
        stages.Add(new TensorRtRuntimeProbeStage("RuntimeCreate", runtimeOk, runtimeMessage));

        return new TensorRtRuntimeProbeReport(line, version, globalRegistry, stages);
    }

    private static IReadOnlyList<TensorRtNativeDependencyInfo> EnumerateNativeBridgeCandidates(List<string> diagnostics)
    {
        List<TensorRtNativeDependencyInfo> results = new List<TensorRtNativeDependencyInfo>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        try
        {
            foreach (string candidate in NativeBridgePathResolver.EnumerateCandidatePaths(typeof(NativeMethodsCommon).Assembly))
            {
                string path = NormalizeProbePath(candidate);
                if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
                {
                    continue;
                }

                results.Add(CreateNativeDependencyInfo(TensorRtNativeDependencySource.NativeBridgeCandidate, path, string.Empty));
            }
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Native bridge candidate enumeration failed", exception);
        }

        return results;
    }

    private static IReadOnlyList<TensorRtNativeDependencyInfo> EnumerateLoadedDependencyModules(List<string> diagnostics)
    {
        List<TensorRtNativeDependencyInfo> results = new List<TensorRtNativeDependencyInfo>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        try
        {
            using Process process = Process.GetCurrentProcess();
            foreach (ProcessModule? module in process.Modules)
            {
                if (module is null)
                {
                    continue;
                }

                string moduleName = SafeGetModuleName(module, diagnostics);
                if (!IsInterestingDependencyName(moduleName))
                {
                    continue;
                }

                string modulePath = NormalizeProbePath(SafeGetModulePath(module, diagnostics));
                string key = !string.IsNullOrWhiteSpace(modulePath) ? modulePath : moduleName;
                if (string.IsNullOrWhiteSpace(key) || !seen.Add(key))
                {
                    continue;
                }

                results.Add(CreateNativeDependencyInfo(TensorRtNativeDependencySource.LoadedProcessModule, modulePath, moduleName));
            }
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Loaded process module enumeration failed", exception);
        }

        return results;
    }

    private static IReadOnlyList<TensorRtNativeDependencyInfo> EnumerateSearchPathDependencyCandidates(TensorRtApiLine line, List<string> diagnostics)
    {
        List<TensorRtNativeDependencyInfo> results = new List<TensorRtNativeDependencyInfo>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        IReadOnlyList<string> patterns = GetDependencySearchPatterns(line);

        foreach (string directory in EnumerateDependencySearchDirectories(diagnostics))
        {
            foreach (string pattern in patterns)
            {
                try
                {
                    foreach (string candidate in Directory.EnumerateFiles(directory, pattern, SearchOption.TopDirectoryOnly))
                    {
                        string path = NormalizeProbePath(candidate);
                        if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
                        {
                            continue;
                        }

                        results.Add(CreateNativeDependencyInfo(TensorRtNativeDependencySource.SearchPathCandidate, path, string.Empty));
                    }
                }
                catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
                {
                    AddDependencyProbeDiagnostic(diagnostics, $"Dependency search failed in '{directory}' for '{pattern}'", exception);
                }
            }
        }

        return results;
    }

    private static IReadOnlyList<string> EnumerateDependencySearchDirectories(List<string> diagnostics)
    {
        List<string> results = new List<string>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void AddDirectory(string? directory)
        {
            string path = NormalizeProbePath(directory ?? string.Empty);
            if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
            {
                return;
            }

            try
            {
                if (Directory.Exists(path))
                {
                    results.Add(path);
                }
            }
            catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
            {
                AddDependencyProbeDiagnostic(diagnostics, $"Dependency directory check failed for '{path}'", exception);
            }
        }

        string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (string entry in currentPath.Split(new[] { Path.PathSeparator }, StringSplitOptions.RemoveEmptyEntries))
        {
            AddDirectory(entry);
        }

        try
        {
            foreach (string entry in NativeBridgePathResolver.EnumerateDependencyDirectories(typeof(NativeMethodsCommon).Assembly))
            {
                AddDirectory(entry);
            }
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Dependency directory resolver enumeration failed", exception);
        }

        return results;
    }

    private static IReadOnlyList<string> GetDependencySearchPatterns(TensorRtApiLine line)
    {
        List<string> patterns = new List<string>();
        if (IsWindowsDependencyProbe())
        {
            switch (line)
            {
                case TensorRtApiLine.TensorRt10:
                    patterns.Add("nvinfer_10.dll");
                    patterns.Add("nvinfer_plugin_10.dll");
                    patterns.Add("nvonnxparser_10.dll");
                    break;
                case TensorRtApiLine.TensorRt11:
                    patterns.Add("nvinfer_11.dll");
                    patterns.Add("nvinfer_plugin_11.dll");
                    patterns.Add("nvonnxparser_11.dll");
                    break;
                default:
                    patterns.Add("nvinfer.dll");
                    patterns.Add("nvinfer_plugin.dll");
                    patterns.Add("nvonnxparser.dll");
                    break;
            }

            patterns.Add("cudart64_*.dll");
            patterns.Add("cudnn*.dll");
            patterns.Add("nvcuda.dll");
            patterns.Add(NativeBridgePathResolver.GetBridgeFileName());
            return patterns;
        }

        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                patterns.Add("libnvinfer.so.10*");
                patterns.Add("libnvinfer_plugin.so.10*");
                patterns.Add("libnvonnxparser.so.10*");
                break;
            case TensorRtApiLine.TensorRt11:
                patterns.Add("libnvinfer.so.11*");
                patterns.Add("libnvinfer_plugin.so.11*");
                patterns.Add("libnvonnxparser.so.11*");
                break;
            default:
                patterns.Add("libnvinfer.so*");
                patterns.Add("libnvinfer_plugin.so*");
                patterns.Add("libnvonnxparser.so*");
                break;
        }

        patterns.Add("libcudart.so*");
        patterns.Add("libcudnn.so*");
        patterns.Add("libcuda.so*");
        patterns.Add(NativeBridgePathResolver.GetBridgeFileName());
        return patterns;
    }

    private static bool IsWindowsDependencyProbe()
    {
#if JYPPX_NETFRAMEWORK
        PlatformID platform = Environment.OSVersion.Platform;
        return platform == PlatformID.Win32NT
            || platform == PlatformID.Win32S
            || platform == PlatformID.Win32Windows
            || platform == PlatformID.WinCE;
#else
        return RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
#endif
    }

    private static TensorRtNativeDependencyInfo CreateNativeDependencyInfo(TensorRtNativeDependencySource source, string path, string name)
    {
        string moduleName = !string.IsNullOrWhiteSpace(name) ? name : Path.GetFileName(path);
        bool exists = false;
        string fileVersion = string.Empty;
        string productVersion = string.Empty;
        string diagnostic = string.Empty;

        try
        {
            exists = !string.IsNullOrWhiteSpace(path) && File.Exists(path);
        }
        catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
        {
            diagnostic = $"Existence check failed: {exception.Message}";
        }

        if (exists)
        {
            try
            {
                FileVersionInfo versionInfo = FileVersionInfo.GetVersionInfo(path);
                fileVersion = versionInfo.FileVersion ?? string.Empty;
                productVersion = versionInfo.ProductVersion ?? string.Empty;
            }
            catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
            {
                diagnostic = $"Version metadata read failed: {exception.Message}";
            }
        }

        return new TensorRtNativeDependencyInfo(source, moduleName, path, exists, fileVersion, productVersion, diagnostic);
    }

    private static bool IsInterestingDependencyName(string moduleName)
    {
        if (string.IsNullOrWhiteSpace(moduleName))
        {
            return false;
        }

        string lower = moduleName.ToLowerInvariant();
        return lower.Contains("jyppxtrtbridge") ||
               lower.Contains("nvinfer") ||
               lower.Contains("nvonnxparser") ||
               lower.Contains("cudart") ||
               lower.Contains("cudnn") ||
               lower.Contains("nvcuda") ||
               lower.Contains("cuda");
    }

    private static string NormalizeProbePath(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        try
        {
            return Path.GetFullPath(value.Trim().Trim('"'));
        }
        catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
        {
            return value.Trim().Trim('"');
        }
    }

    private static string SafeGetModuleName(ProcessModule module, List<string> diagnostics)
    {
        try
        {
            return module.ModuleName ?? string.Empty;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Process module name read failed", exception);
            return string.Empty;
        }
    }

    private static string SafeGetModulePath(ProcessModule module, List<string> diagnostics)
    {
        try
        {
            return module.FileName ?? string.Empty;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Process module path read failed", exception);
            return string.Empty;
        }
    }

    private static void AddDependencyProbeDiagnostic(List<string> diagnostics, string stage, Exception exception)
    {
        if (diagnostics.Count >= 64)
        {
            return;
        }

        diagnostics.Add($"{stage}: {exception.Message}");
    }

    private static TensorRtAdapterInfo GetAdapterInfoOrFallback(TensorRtApiLine line, BridgeBuildInfo buildInfo)
    {
        try
        {
            return BridgeInfoMapper.ToManaged(NativeBridgeApi.GetAdapterInfo(line));
        }
        catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            return new TensorRtAdapterInfo(
                line,
                buildInfo.HasTensorRt,
                runtimeCreationSupported: false,
                builderCreationSupported: false,
                networkCreationSupported: false,
                engineDeserializationSupported: false,
                buildInfo.TensorRtVersion,
                exception.Message);
        }
    }

}
