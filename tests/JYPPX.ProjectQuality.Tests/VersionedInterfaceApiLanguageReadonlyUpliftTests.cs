using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class VersionedInterfaceApiLanguageReadonlyUpliftTests
{
    [Fact]
    public void ApiLanguageManifestsOnlyPromoteConcreteOwnedCallbackObjects()
    {
        string trt10InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-callback-interface-info.manifest.json");
        string trt11InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-callback-interface-info.manifest.json");
        string trt10DeferredManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-third-batch-other-deferred.manifest.json");
        string trt11DeferredManifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");

        Assert.Contains("trt10-progress-monitor-get-api-language", trt10InterfaceManifest);
        Assert.DoesNotContain("trt10-logger-get-api-language", trt10InterfaceManifest);
        Assert.DoesNotContain("trt10-profiler-get-api-language", trt10InterfaceManifest);

        Assert.Contains("trt11-logger-get-api-language", trt11InterfaceManifest);
        Assert.Contains("trt11-profiler-get-api-language", trt11InterfaceManifest);
        Assert.Contains("trt11-progress-monitor-get-api-language", trt11InterfaceManifest);

        Assert.Contains("out_api_language", trt10InterfaceManifest);
        Assert.Contains("\"managedType\": \"out int\"", trt10InterfaceManifest);
        Assert.Contains("out_api_language", trt11InterfaceManifest);
        Assert.Contains("\"managedType\": \"out int\"", trt11InterfaceManifest);

        Assert.Contains("trt10-versioned-interface-get-api-language-deferred", trt10DeferredManifest);
        Assert.Contains("trt11-versioned-interface-get-api-language-deferred", trt11DeferredManifest);
    }

    [Fact]
    public void NativeApiLanguageQueriesUseScalarCopyAndKeepRawBaseDeferred()
    {
        string trt10NativeSource = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11NativeSource = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string trt10DeferredSource = ReadSource("native", "src", "tensorrt", "v10", "modules", "deferred", "cross_version_other_deferred.inc");
        string trt11DeferredSource = ReadSource("native", "src", "tensorrt", "v11", "modules", "deferred", "twenty_third_batch_deferred.inc");

        Assert.Contains("jyppx_trt10_progress_monitor_get_api_language", trt10NativeSource);
        Assert.Contains("get_callback_api_language(monitor_payload", trt10NativeSource);
        Assert.Contains("static_cast<int32_t>(object->getAPILanguage())", trt10NativeSource);
        Assert.DoesNotContain("jyppx_trt10_logger_get_api_language", trt10NativeSource);
        Assert.DoesNotContain("jyppx_trt10_profiler_get_api_language", trt10NativeSource);

        Assert.Contains("jyppx_trt11_logger_get_api_language", trt11NativeSource);
        Assert.Contains("jyppx_trt11_profiler_get_api_language", trt11NativeSource);
        Assert.Contains("jyppx_trt11_progress_monitor_get_api_language", trt11NativeSource);
        Assert.Contains("get_callback_api_language(logger_payload", trt11NativeSource);
        Assert.Contains("get_callback_api_language(profiler_payload", trt11NativeSource);
        Assert.Contains("get_callback_api_language(monitor_payload", trt11NativeSource);

        Assert.Contains("jyppx_trt10_versioned_interface_get_api_language_deferred", trt10DeferredSource);
        Assert.Contains("jyppx_trt11_versioned_interface_get_api_language_deferred", trt11DeferredSource);
        Assert.Contains("raw base pointer ABI", trt11DeferredSource);
    }

    [Fact]
    public void ManagedApiLanguageWrappersExposeEnumAndNoRawPointers()
    {
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Callbacks", "NativeBridgeApi.CallbackInterfaceInfo.cs");
        string interfaceInfoSource = ReadSource("src", "JYPPX.TensorRtSharp", "Interfaces", "TensorRtInterfaceInfo.cs");
        string loggerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtLogger.cs");
        string profilerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfiler.cs");
        string monitorSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProgressMonitor.cs");

        Assert.Contains("public enum TensorRtApiLanguage", interfaceInfoSource);
        Assert.Contains("Unknown = -1", interfaceInfoSource);
        Assert.Contains("Cpp = 0", interfaceInfoSource);
        Assert.Contains("Python = 1", interfaceInfoSource);

        Assert.Contains("GetLoggerApiLanguage", interopSource);
        Assert.Contains("GetProfilerApiLanguage", interopSource);
        Assert.Contains("GetProgressMonitorApiLanguage", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_progress_monitor_get_api_language", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_logger_get_api_language", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_profiler_get_api_language", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_progress_monitor_get_api_language", interopSource);

        Assert.Contains("public TensorRtApiLanguage ApiLanguage", loggerSource);
        Assert.Contains("public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)", loggerSource);
        Assert.Contains("public TensorRtApiLanguage ApiLanguage", profilerSource);
        Assert.Contains("public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)", profilerSource);
        Assert.Contains("public TensorRtApiLanguage ApiLanguage", monitorSource);
        Assert.Contains("public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)", monitorSource);

        string publicSources = string.Join(Environment.NewLine, loggerSource, profilerSource, monitorSource);
        Assert.DoesNotContain("public IntPtr", publicSources);
        Assert.DoesNotContain("public nint", publicSources);
    }

    [Fact]
    public void SmokeRunnersPrintApiLanguageProbeResults()
    {
        string loggerSmoke = ReadSource("smoke", "ManagedLoggerCallbackSmokeRunner", "Program.cs");
        string profilerSmoke = ReadSource("smoke", "ManagedProfilerCallbackSmokeRunner", "Program.cs");
        string monitorSmoke = ReadSource("smoke", "ManagedProgressMonitorSmokeRunner", "Program.cs");

        Assert.Contains("logger.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage", loggerSmoke);
        Assert.Contains("ManagedLoggerApiLanguage", loggerSmoke);
        Assert.Contains("profiler.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage", profilerSmoke);
        Assert.Contains("ManagedProfilerApiLanguage", profilerSmoke);
        Assert.Contains("monitor.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage", monitorSmoke);
        Assert.Contains("ManagedProgressMonitorApiLanguage", monitorSmoke);
    }

    [Fact]
    public void GeneratedBindingsCarryApiLanguageEntrypoints()
    {
        string nativeMethods = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string entryPointNames = ReadSource("src", "JYPPX.Shared", "Generated", "GeneratedEntryPointNames.g.cs");
        string trt10Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string trt11Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");

        Assert.Contains("jyppx_trt10_progress_monitor_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt11_logger_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt11_profiler_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt11_progress_monitor_get_api_language", nativeMethods);
        Assert.Contains("out int out_api_language", nativeMethods);

        Assert.Contains("Trt10ProgressMonitorGetApiLanguage", entryPointNames);
        Assert.Contains("Trt11LoggerGetApiLanguage", entryPointNames);
        Assert.Contains("Trt11ProfilerGetApiLanguage", entryPointNames);
        Assert.Contains("Trt11ProgressMonitorGetApiLanguage", entryPointNames);

        Assert.Contains("jyppx_trt10_progress_monitor_get_api_language", trt10Header);
        Assert.Contains("jyppx_trt11_logger_get_api_language", trt11Header);
        Assert.Contains("jyppx_trt11_profiler_get_api_language", trt11Header);
        Assert.Contains("jyppx_trt11_progress_monitor_get_api_language", trt11Header);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
