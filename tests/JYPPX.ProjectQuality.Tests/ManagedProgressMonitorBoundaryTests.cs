using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedProgressMonitorBoundaryTests
{
    [Fact]
    public void ManagedProgressMonitorApisAreManifestedForTensorRt10And11()
    {
        string trt10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-minimal.manifest.json");
        string trt11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-minimal.manifest.json");
        string trt8Deferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");

        Assert.Contains("trt10-progress-monitor-create-with-callback", trt10);
        Assert.Contains("trt10-progress-monitor-emit-diagnostic", trt10);
        Assert.Contains("trt10-builder-config-set-progress-monitor", trt10);
        Assert.Contains("trt10-builder-config-has-progress-monitor", trt10);
        Assert.Contains("trt10-builder-config-clear-progress-monitor", trt10);

        Assert.Contains("trt11-progress-monitor-create-with-callback", trt11);
        Assert.Contains("trt11-progress-monitor-emit-diagnostic", trt11);
        Assert.Contains("trt11-builder-config-set-progress-monitor", trt11);
        Assert.Contains("JYPPX_TensorRtProgressMonitorCallback", trt10);
        Assert.Contains("JYPPX_TensorRtProgressMonitorCallback", trt11);
        Assert.DoesNotContain("progress-monitor-create-with-callback", trt8Deferred);
    }

    [Fact]
    public void ManagedProgressMonitorInterfaceInfoIsReadOnlyForTensorRt10And11()
    {
        string trt10InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-callback-interface-info.manifest.json");
        string trt11InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-callback-interface-info.manifest.json");
        string trt10NativeSource = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11NativeSource = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.CallbackInterfaceInfo.cs");
        string monitorSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProgressMonitor.cs");
        string interfaceInfoSource = ReadSource("src", "JYPPX.TensorRtSharp", "Interfaces", "TensorRtInterfaceInfo.cs");

        Assert.Contains("trt10-progress-monitor-get-interface-info", trt10InterfaceManifest);
        Assert.Contains("trt11-progress-monitor-get-interface-info", trt11InterfaceManifest);
        Assert.Contains("\"managedType\": \"byte[]\"", trt10InterfaceManifest);
        Assert.Contains("\"managedType\": \"byte[]\"", trt11InterfaceManifest);
        Assert.Contains("out_required_size", trt10InterfaceManifest);
        Assert.Contains("out_required_size", trt11InterfaceManifest);
        Assert.Contains("out_major", trt10InterfaceManifest);
        Assert.Contains("out_minor", trt11InterfaceManifest);

        Assert.Contains("jyppx_trt10_progress_monitor_get_interface_info", trt10NativeSource);
        Assert.Contains("jyppx_trt11_progress_monitor_get_interface_info", trt11NativeSource);
        Assert.Contains("get_callback_interface_info(monitor_payload", trt10NativeSource);
        Assert.Contains("get_callback_interface_info(monitor_payload", trt11NativeSource);
        Assert.Contains("copy_interface_info_to_buffer", trt10NativeSource);
        Assert.Contains("copy_interface_info_to_buffer", trt11NativeSource);

        Assert.Contains("GetProgressMonitorInterfaceInfo", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_progress_monitor_get_interface_info", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_progress_monitor_get_interface_info", interopSource);

        Assert.Contains("public TensorRtInterfaceInfo InterfaceInfo", monitorSource);
        Assert.Contains("public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)", monitorSource);
        Assert.Contains("public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)", monitorSource);
        Assert.Contains("Unsupported or unavailable adapters return", monitorSource);
        Assert.Contains("public readonly struct TensorRtInterfaceInfo", interfaceInfoSource);
        Assert.Contains("public string Kind", interfaceInfoSource);
        Assert.DoesNotContain("public IntPtr", monitorSource);
        Assert.DoesNotContain("public nint", monitorSource);
    }

    [Fact]
    public void ManagedProgressMonitorPublicApiOwnsCallbackStateWithoutExposingNativePointers()
    {
        string monitorSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProgressMonitor.cs");
        string configSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.cs");
        string diagnosticsSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11Diagnostics.cs");

        Assert.Contains("public delegate bool TensorRtProgressMonitorHandler", monitorSource);
        Assert.Contains("public sealed class TensorRtProgressMonitor", monitorSource);
        Assert.Contains("GCHandle.Alloc(_callbackState)", monitorSource);
        Assert.Contains("~TensorRtProgressMonitor()", monitorSource);
        Assert.Contains("AttachBorrower", monitorSource);
        Assert.Contains("DetachBorrower", monitorSource);
        Assert.Contains("LastCallbackException", monitorSource);
        Assert.Contains("they never cross the native ABI boundary", monitorSource);
        Assert.DoesNotContain("public IntPtr", monitorSource);
        Assert.DoesNotContain("public nint", monitorSource);

        Assert.Contains("_progressMonitorKeepAlive", configSource);
        Assert.Contains("DetachProgressMonitor();", configSource);
        Assert.Contains("public void SetProgressMonitor(TensorRtProgressMonitor monitor)", diagnosticsSource);
        Assert.Contains("NativeBridgeApi.SetBuilderConfigProgressMonitor", diagnosticsSource);
    }

    [Fact]
    public void BuilderConfigDisposeKeepsBorrowedProgressMonitorAliveThroughNativeConfigRelease()
    {
        string configSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.cs");

        int clearIndex = configSource.IndexOf("TryClearProgressMonitorForDispose();", StringComparison.Ordinal);
        int disposeIndex = configSource.IndexOf("_handle.Dispose();", StringComparison.Ordinal);
        int keepAliveIndex = configSource.IndexOf("GC.KeepAlive(monitor);", StringComparison.Ordinal);
        int detachIndex = configSource.IndexOf("DetachProgressMonitor();", StringComparison.Ordinal);

        Assert.True(clearIndex >= 0, "Dispose should first try to clear the native borrowed progress monitor pointer.");
        Assert.True(disposeIndex > clearIndex, "The native builder-config handle must be released after the native clear attempt.");
        Assert.True(keepAliveIndex > disposeIndex, "The progress monitor must be kept alive through native config release.");
        Assert.True(detachIndex > keepAliveIndex, "The managed borrow count must be detached only after the config handle is released.");
        Assert.Contains("catch (BridgeProbeException)", configSource);
        Assert.Contains("TensorRT never observes a freed borrowed monitor", configSource);
    }

    [Fact]
    public void NativeManagedProgressMonitorTrampolineSwallowsCallbackFailures()
    {
        string nativeSources = string.Join(
            Environment.NewLine,
            ReadSource("native", "src", "tensorrt", "v10", "api.cpp"),
            ReadSource("native", "src", "tensorrt", "v11", "api.cpp"));

        Assert.Contains("class ManagedProgressMonitor final", nativeSources);
        Assert.Contains("JYPPX_TensorRtProgressMonitorCallback callback_", nativeSources);
        Assert.Contains("callback_status != JYPPX_STATUS_OK", nativeSources);
        Assert.Contains("last_callback_failed_ = true", nativeSources);
        Assert.Contains("Managed TensorRT progress monitor callback returned status", nativeSources);
        Assert.Contains("catch (const std::exception& exception)", nativeSources);
        Assert.Contains("catch (...)", nativeSources);
        Assert.Contains("*continue_output = JYPPX_TRUE", nativeSources);
    }

    [Fact]
    public void ManagedProgressMonitorSmokeProvidesSkippableConsumerCoverage()
    {
        string project = ReadSource("smoke", "ManagedProgressMonitorSmokeRunner", "ManagedProgressMonitorSmokeRunner.csproj");
        string program = ReadSource("smoke", "ManagedProgressMonitorSmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");

        Assert.Contains("ManagedProgressMonitorSmokeRunner", smokeReadme);
        Assert.Contains("ManagedProgressMonitorSmokeRunner.csproj", solution);
        Assert.Contains("<ProjectReference Include=\"..\\..\\src\\JYPPX.TensorRtSharp\\JYPPX.TensorRtSharp.csproj\" />", project);

        Assert.Contains("--dependency-probe-only", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
        Assert.Contains("new TensorRtProgressMonitor(", program);
        Assert.Contains("EmitDiagnostic", program);
        Assert.Contains("monitor.TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)", program);
        Assert.Contains("ManagedProgressMonitorInterfaceInfo Skipped=True Reason={diagnostic}", program);
        Assert.Contains("ManagedProgressMonitorInterfaceInfo", program);
        Assert.Contains("ManagedProgressMonitorException", program);
        Assert.Contains("SetProgressMonitor", program);
        Assert.Contains("ClearProgressMonitor", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
