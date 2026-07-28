using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedLoggerCallbackBoundaryTests
{
    [Fact]
    public void ManagedLoggerCallbackApisAreManifestedForAllTensorRtLines()
    {
        string trt8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-minimal.manifest.json");
        string trt10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-minimal.manifest.json");
        string trt11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-minimal.manifest.json");

        Assert.Contains("trt8-logger-create-with-callback", trt8);
        Assert.Contains("trt8-logger-emit-diagnostic", trt8);
        Assert.Contains("trt10-logger-create-with-callback", trt10);
        Assert.Contains("trt10-logger-emit-diagnostic", trt10);
        Assert.Contains("trt11-logger-create-with-callback", trt11);
        Assert.Contains("trt11-logger-emit-diagnostic", trt11);

        Assert.Contains("JYPPX_TensorRtLoggerCallback", trt8);
        Assert.Contains("JYPPX_TensorRtLoggerCallback", trt10);
        Assert.Contains("JYPPX_TensorRtLoggerCallback", trt11);
    }

    [Fact]
    public void ManagedLoggerInterfaceInfoIsReadOnlyAndTensorRt11Only()
    {
        string trt10InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-callback-interface-info.manifest.json");
        string trt11InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-callback-interface-info.manifest.json");
        string nativeSource = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.CallbackInterfaceInfo.cs");
        string loggerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtLogger.cs");

        Assert.DoesNotContain("trt10-logger-get-interface-info", trt10InterfaceManifest);
        Assert.Contains("trt11-logger-get-interface-info", trt11InterfaceManifest);
        Assert.Contains("\"managedType\": \"byte[]\"", trt11InterfaceManifest);
        Assert.Contains("out_required_size", trt11InterfaceManifest);
        Assert.Contains("out_major", trt11InterfaceManifest);
        Assert.Contains("out_minor", trt11InterfaceManifest);

        Assert.Contains("jyppx_trt11_logger_get_interface_info", nativeSource);
        Assert.Contains("copy_interface_info_to_buffer", nativeSource);
        Assert.Contains("get_callback_interface_info(logger_payload", nativeSource);

        Assert.Contains("GetLoggerInterfaceInfo", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_logger_get_interface_info", interopSource);
        Assert.Contains("throw UnsupportedCallbackInterfaceInfoLine", interopSource);

        Assert.Contains("public TensorRtInterfaceInfo InterfaceInfo", loggerSource);
        Assert.Contains("public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)", loggerSource);
        Assert.Contains("public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)", loggerSource);
        Assert.Contains("TensorRT 8 and TensorRT 10 loggers are not versioned interfaces", loggerSource);
        Assert.DoesNotContain("public IntPtr", loggerSource);
        Assert.DoesNotContain("public nint", loggerSource);
    }

    [Fact]
    public void ManagedLoggerPublicApiOwnsCallbackStateWithoutExposingNativePointers()
    {
        string loggerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtLogger.cs");
        string runtimeSource = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.cs");
        string builderSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.cs");
        string parserSource = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs");
        string refitterSource = ReadSource("src", "JYPPX.TensorRtSharp", "Refit", "TensorRtRefitter.cs");

        Assert.Contains("public delegate void TensorRtLogHandler", loggerSource);
        Assert.Contains("public TensorRtLogger(TensorRtApiLine line, TensorRtLogHandler handler", loggerSource);
        Assert.Contains("public bool EmitDiagnostic", loggerSource);
        Assert.Contains("GCHandle.Alloc(_callbackState)", loggerSource);
        Assert.Contains("FreeCallbackState();", loggerSource);
        Assert.Contains("~TensorRtLogger()", loggerSource);
        Assert.Contains("CallbackFailureCount", loggerSource);
        Assert.Contains("LastCallbackException", loggerSource);
        Assert.Contains("AttachBorrower", loggerSource);
        Assert.Contains("DetachBorrower", loggerSource);
        Assert.Contains("ReleaseHandle", loggerSource);
        Assert.Contains("public bool IsAttached", loggerSource);
        Assert.Contains("they are never allowed to cross the native ABI boundary", loggerSource);
        Assert.DoesNotContain("public IntPtr", loggerSource);
        Assert.DoesNotContain("public nint", loggerSource);

        Assert.Contains("_loggerKeepAlive", runtimeSource);
        Assert.Contains("_loggerKeepAlive.AttachBorrower(Line)", runtimeSource);
        Assert.Contains("_loggerKeepAlive.DetachBorrower()", runtimeSource);
        Assert.Contains("private bool _disposed", runtimeSource);
        Assert.Contains("_loggerKeepAlive", builderSource);
        Assert.Contains("_loggerKeepAlive.AttachBorrower(Line)", builderSource);
        Assert.Contains("_loggerKeepAlive.DetachBorrower()", builderSource);
        Assert.Contains("private bool _disposed", builderSource);
        Assert.Contains("_loggerKeepAlive", parserSource);
        Assert.Contains("_loggerKeepAlive.AttachBorrower(Line)", parserSource);
        Assert.Contains("_loggerKeepAlive?.DetachBorrower()", parserSource);
        Assert.Contains("private bool _disposed", parserSource);
        Assert.Contains("_loggerKeepAlive", refitterSource);
        Assert.Contains("_loggerKeepAlive?.DetachBorrower()", refitterSource);
        Assert.Contains("private bool _disposeRequested", refitterSource);
        Assert.Contains("private bool _handleReleased", refitterSource);
        Assert.Contains("private int _attachmentCount", refitterSource);
    }

    [Fact]
    public void ManagedLoggerBorrowersAttachBeforeNativeCallsAndDetachOnDispose()
    {
        string engineSource = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");
        string refitterSource = ReadSource("src", "JYPPX.TensorRtSharp", "Refit", "TensorRtRefitter.cs");
        string runtimeSource = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.cs");
        string builderSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.cs");
        string parserSource = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs");

        Assert.Contains("logger.AttachBorrower(Line);", engineSource);
        Assert.Contains("NativeBridgeApi.CreateRefitter(Line, _handle, logger.Handle)", engineSource);
        Assert.Contains("logger.DetachBorrower();", engineSource);
        Assert.Contains("loggerBorrowAttached: true", engineSource);
        Assert.Contains("loggerBorrowAttached = false", refitterSource);

        Assert.Contains("_loggerKeepAlive.AttachBorrower(Line);", runtimeSource);
        Assert.Contains("_loggerKeepAlive.DetachBorrower();", runtimeSource);
        Assert.Contains("_loggerKeepAlive.AttachBorrower(Line);", builderSource);
        Assert.Contains("_loggerKeepAlive.DetachBorrower();", builderSource);
        Assert.Contains("_loggerKeepAlive.AttachBorrower(Line);", parserSource);
        Assert.Contains("_loggerKeepAlive?.DetachBorrower();", parserSource);

        Assert.Contains("TensorRT borrows the logger pointer", runtimeSource);
        Assert.Contains("TensorRT borrows the logger pointer", builderSource);
        Assert.Contains("TensorRT borrows the logger pointer", parserSource);
    }

    [Fact]
    public void NativeManagedLoggerTrampolineSwallowsCallbackFailures()
    {
        string nativeSources = string.Join(
            Environment.NewLine,
            ReadSource("native", "src", "tensorrt", "v8", "api.cpp"),
            ReadSource("native", "src", "tensorrt", "v10", "api.cpp"),
            ReadSource("native", "src", "tensorrt", "v11", "api.cpp"));

        Assert.Contains("JYPPX_TensorRtLoggerCallback callback_", nativeSources);
        Assert.Contains("callback_status != JYPPX_STATUS_OK", nativeSources);
        Assert.Contains("last_callback_failed_ = true", nativeSources);
        Assert.Contains("catch (const std::exception& exception)", nativeSources);
        Assert.Contains("catch (...)", nativeSources);
        Assert.Contains("Managed TensorRT logger callback returned status", nativeSources);
    }

    [Fact]
    public void ManagedLoggerCallbackSmokeProvidesSkippableConsumerCoverage()
    {
        string project = ReadSource("smoke", "ManagedLoggerCallbackSmokeRunner", "ManagedLoggerCallbackSmokeRunner.csproj");
        string program = ReadSource("smoke", "ManagedLoggerCallbackSmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");

        Assert.Contains("ManagedLoggerCallbackSmokeRunner", smokeReadme);
        Assert.Contains("ManagedLoggerCallbackSmokeRunner.csproj", solution);
        Assert.Contains("<ProjectReference Include=\"..\\..\\src\\JYPPX.TensorRtSharp\\JYPPX.TensorRtSharp.csproj\" />", project);

        Assert.Contains("--dependency-probe-only", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
        Assert.Contains("Skipped=True Reason=AdapterNotReady", program);
        Assert.Contains("new TensorRtLogger(", program);
        Assert.Contains("EmitDiagnostic", program);
        Assert.Contains("logger.TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)", program);
        Assert.Contains("ManagedLoggerInterfaceInfo Skipped=True Reason={diagnostic}", program);
        Assert.Contains("ManagedLoggerInterfaceInfo", program);
        Assert.Contains("ManagedLoggerCallbackException", program);
        Assert.Contains("LastCallbackException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
