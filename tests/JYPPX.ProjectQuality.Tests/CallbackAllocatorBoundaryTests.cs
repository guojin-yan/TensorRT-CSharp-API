using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CallbackAllocatorBoundaryTests
{
    [Fact]
    public void UnsafeCallbackAndAllocatorRowsRemainExplicitDeferred()
    {
        string trt11CallbackDeferred = ReadTensorRtManifest("v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");
        string trt10OtherDeferred = ReadTensorRtManifest("v10", "trt10-cross-version-third-batch-other-deferred.manifest.json");
        string trt10DiagnosticsDeferred = ReadTensorRtManifest("v10", "trt10-cross-version-sixth-batch-diagnostics-refitter-deferred.manifest.json");
        string trt8CallbackDeferred = ReadTensorRtManifest("v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string trt8OtherDeferred = ReadTensorRtManifest("v8", "trt8-cross-version-tenth-batch-other-deferred.manifest.json");

        Assert.Contains("trt11-error-recorder-get-error-code-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-logger-finder-find-logger-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-profiler-report-layer-time-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-gpu-allocator-allocate-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-output-allocator-reallocate-output-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-progress-monitor-step-complete-deferred", trt11CallbackDeferred);

        Assert.Contains("trt10-gpu-allocator-allocate-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-output-allocator-reallocate-output-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-progress-monitor-step-complete-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-error-recorder-get-nb-errors-deferred", trt10DiagnosticsDeferred);
        Assert.Contains("trt10-profiler-report-layer-time-deferred", trt10DiagnosticsDeferred);

        Assert.Contains("trt8-error-recorder-get-nb-errors-deferred", trt8CallbackDeferred);
        Assert.Contains("trt8-profiler-report-layer-time-deferred", trt8CallbackDeferred);
        Assert.Contains("trt8-gpu-allocator-allocate-deferred", trt8OtherDeferred);
        Assert.Contains("trt8-output-allocator-reallocate-output-deferred", trt8OtherDeferred);
    }

    [Fact]
    public void AllocatorCallbackOwnerDesignGateDocumentsDeferredRequirements()
    {
        string design = ReadSource("docs", "articles", "zh-cn", "allocator-callback-owner-design.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latestStatus = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string trt8OtherDeferred = ReadTensorRtManifest("v8", "trt8-cross-version-tenth-batch-other-deferred.manifest.json");
        string trt10OtherDeferred = ReadTensorRtManifest("v10", "trt10-cross-version-third-batch-other-deferred.manifest.json");
        string trt11CallbackDeferred = ReadTensorRtManifest("v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");

        Assert.Contains("Allocator Owner 与 Callback Trampoline 设计门禁", design);
        Assert.Contains("native owner", design);
        Assert.Contains("Managed owner", design);
        Assert.Contains("delegate pinning", design);
        Assert.Contains("device pointer ownership", design);
        Assert.Contains("Dry-run owner skeleton 进展", design);
        Assert.Contains("TensorRtAllocatorCallbackOwner", design);
        Assert.Contains("allocator-owner-dry-run-diagnostics", design);
        Assert.Contains("不能作为 `IGpuAllocator::allocate/free/deallocate/reallocate`", design);
        Assert.Contains("Native owner 与 ledger 设计进展", design);
        Assert.Contains("allocator-owner-ledger-design.md", design);
        Assert.Contains("allocator-owner-ledger-design-gate", design);
        Assert.Contains("不能作为真实 TensorRT allocator callback 已启用的证据", design);
        Assert.Contains("ABI no-throw", design);
        Assert.Contains("Stream 与 async 边界", design);
        Assert.Contains("TRT8、TRT10、TRT11", design);
        Assert.Contains("IGpuAllocator::allocate", design);
        Assert.Contains("IGpuAsyncAllocator::allocateAsync", design);
        Assert.Contains("IOutputAllocator::reallocateOutput", design);
        Assert.Contains("IDebugListener::processDebugTensor", design);
        Assert.Contains("继续 deferred", design);

        Assert.Contains("allocator-callback-owner-design.md", docsIndex);
        Assert.Contains("allocator-callback-owner-design.md", docsToc);
        Assert.Contains("allocator-owner-ledger-design.md", docsIndex);
        Assert.Contains("allocator-owner-ledger-design.md", docsToc);
        Assert.Contains("allocator-owner-ledger-design.md", latestStatus);
        Assert.Contains("Allocator Owner 与 Callback Trampoline 设计门禁", latestStatus);

        Assert.Contains("trt8-gpu-allocator-allocate-deferred", trt8OtherDeferred);
        Assert.Contains("trt8-gpu-allocator-free-deferred", trt8OtherDeferred);
        Assert.Contains("trt8-output-allocator-notify-shape-deferred", trt8OtherDeferred);
        Assert.Contains("trt8-output-allocator-reallocate-output-deferred", trt8OtherDeferred);
        Assert.Contains("trt10-gpu-allocator-allocate-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-gpu-allocator-deallocate-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-gpu-async-allocator-allocate-async-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-output-allocator-notify-shape-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-output-allocator-reallocate-output-deferred", trt10OtherDeferred);
        Assert.Contains("trt10-debug-listener-process-debug-tensor-deferred", trt10OtherDeferred);
        Assert.Contains("trt11-gpu-allocator-allocate-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-gpu-allocator-deallocate-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-gpu-async-allocator-allocate-async-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-output-allocator-notify-shape-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-output-allocator-reallocate-output-deferred", trt11CallbackDeferred);
        Assert.Contains("trt11-debug-listener-process-debug-tensor-deferred", trt11CallbackDeferred);

        Assert.Contains("\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAsyncAllocator\",\"allocateAsync\",\"IGpuAsyncAllocator::allocateAsync\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void AllocatorOwnerLedgerDesignGateDocumentsNativeOwnerLedgerAndStillDefersCallbacks()
    {
        string ledgerDesign = ReadSource("docs", "articles", "zh-cn", "allocator-owner-ledger-design.md");
        string callbackDesign = ReadSource("docs", "articles", "zh-cn", "allocator-callback-owner-design.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latestStatus = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("Native Allocator Owner 与 Device Pointer Ledger 设计门禁", ledgerDesign);
        Assert.Contains("allocator-owner-ledger-design-gate", ledgerDesign);
        Assert.Contains("Native Owner 形状", ledgerDesign);
        Assert.Contains("Device Pointer Ledger", ledgerDesign);
        Assert.Contains("失败与异常映射", ledgerDesign);
        Assert.Contains("跨版本 Route", ledgerDesign);
        Assert.Contains("Public C# 边界", ledgerDesign);
        Assert.Contains("create/destroy/release hook", ledgerDesign);
        Assert.Contains("owner 不可复制", ledgerDesign);
        Assert.Contains("stable address", ledgerDesign);
        Assert.Contains("no-throw", ledgerDesign);
        Assert.Contains("pointerValue", ledgerDesign);
        Assert.Contains("allocationId", ledgerDesign);
        Assert.Contains("跨 owner 释放", ledgerDesign);
        Assert.Contains("Windows SEH", ledgerDesign);
        Assert.Contains("TRT8 使用 `IGpuAllocator::free`", ledgerDesign);
        Assert.Contains("TRT10/TRT11 的 `IGpuAllocator::deallocate`", ledgerDesign);
        Assert.Contains("不能作为真实 TensorRT allocator callback 已启用的证据", callbackDesign);

        Assert.Contains("allocator-owner-ledger-design.md", docsIndex);
        Assert.Contains("allocator-owner-ledger-design.md", docsToc);
        Assert.Contains("allocator-owner-ledger-design.md", latestStatus);
        Assert.Contains("allocator-owner-ledger-design.md", callbackDesign);

        Assert.Contains("New-AllocatorOwnerLedgerDesignGateEvidence", readiness);
        Assert.Contains("allocatorOwnerLedgerDesignGate", readiness);
        Assert.Contains("hasDeferredRowEvidence", readiness);
        Assert.Contains("missingDeferredRows", readiness);
        Assert.Contains("allocator owner ledger design gate:", readiness);
        Assert.DoesNotContain("\"allocator-owner-ledger-design-gate\",", bridgeConsumer);

        Assert.Contains("\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAllocator\",\"free\",\"IGpuAllocator::free\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAllocator\",\"deallocate\",\"IGpuAllocator::deallocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAllocator\",\"reallocate\",\"IGpuAllocator::reallocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAsyncAllocator\",\"allocateAsync\",\"IGpuAsyncAllocator::allocateAsync\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void RealCallbackTrampolineGateDocumentsGoNoGoAndKeepsDryRunSeparate()
    {
        string gate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string callbackDesign = ReadSource("docs", "articles", "zh-cn", "allocator-callback-owner-design.md");
        string ledgerDesign = ReadSource("docs", "articles", "zh-cn", "allocator-owner-ledger-design.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latestStatus = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string readme = ReadSource("pack", "runtime-split", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("真实 Callback Trampoline 门禁复审", gate);
        Assert.Contains("real-callback-trampoline-gate", gate);
        Assert.Contains("go/no-go checklist", gate);
        Assert.Contains("native owner 生命周期", gate);
        Assert.Contains("dispose 顺序", gate);
        Assert.Contains("GCHandle", gate);
        Assert.Contains("delegate pinning", gate);
        Assert.Contains("C ABI no-throw", gate);
        Assert.Contains("Windows SEH", gate);
        Assert.Contains("exception-to-status", gate);
        Assert.Contains("device pointer ledger", gate);
        Assert.Contains("stream/async", gate);
        Assert.Contains("real-callback-runtime", gate);
        Assert.Contains("allocator-owner-internal-runtime-prototype", gate);
        Assert.Contains("internal-runtime-prototype", gate);
        Assert.Contains("RealCallbackRuntime=False", gate);
        Assert.Contains("EvidenceKind=allocator-owner-internal-runtime-prototype", gate);
        Assert.Contains("ReleaseHookCount", gate);
        Assert.Contains("CallbackStatePinned", gate);
        Assert.Contains("DelegatePinned", gate);
        Assert.Contains("dry-run", gate);
        Assert.Contains("copied-state", gate);
        Assert.Contains("TRT8", gate);
        Assert.Contains("TRT10", gate);
        Assert.Contains("TRT11", gate);
        Assert.Contains("不能作为真实 TensorRT callback 已启用的证据", gate);
        Assert.Contains("IOutputAllocator::reallocateOutput", gate);
        Assert.Contains("IOutputAllocator::notifyShape", gate);
        Assert.Contains("IDebugListener::processDebugTensor", gate);
        Assert.Contains("IGpuAsyncAllocator::deallocateAsync", gate);

        Assert.Contains("real-callback-trampoline-gate.md", docsIndex);
        Assert.Contains("real-callback-trampoline-gate.md", docsToc);
        Assert.Contains("real-callback-trampoline-gate.md", latestStatus);
        Assert.Contains("real-callback-trampoline-gate.md", callbackDesign);
        Assert.Contains("real-callback-trampoline-gate.md", ledgerDesign);
        Assert.Contains("real-callback-runtime-evidence-schema.md", docsIndex);
        Assert.Contains("real-callback-runtime-evidence-schema.md", docsToc);
        Assert.Contains("real-callback-runtime-evidence-schema.md", latestStatus);
        Assert.Contains("real-callback-runtime-evidence-schema.md", gate);

        Assert.Contains("真实 Callback Runtime Evidence Schema", schema);
        Assert.Contains("real-callback-runtime-evidence-schema", schema);
        Assert.Contains("realCallbackRuntimeEvidenceSchema", schema);
        Assert.Contains("realCallbackRuntimeEvidence", schema);
        Assert.Contains("RealCallbackRuntimeEvidence", schema);
        Assert.Contains("RealCallbackRuntimeEvidence.Status", schema);
        Assert.Contains("blocked-by-cuda-driver", schema);
        Assert.Contains("blocked-by-application-control", schema);
        Assert.Contains("RealCallbackRuntime=True", schema);
        Assert.Contains("EvidenceKind=real-callback-runtime", schema);
        Assert.Contains("CallbackKind", schema);
        Assert.Contains("TensorRtLine", schema);
        Assert.Contains("CudaLine", schema);
        Assert.Contains("RuntimePackageKey", schema);
        Assert.Contains("OwnerId", schema);
        Assert.Contains("InvocationCount", schema);
        Assert.Contains("AllocationCount", schema);
        Assert.Contains("ReleaseCount", schema);
        Assert.Contains("FailureCount", schema);
        Assert.Contains("InFlightCallbackCount", schema);
        Assert.Contains("FullPackageConsumerReport", schema);
        Assert.Contains("allocator-owner-internal-runtime-prototype", schema);
        Assert.Contains("internal-runtime-prototype", schema);
        Assert.Contains("RealCallbackRuntime=False", schema);
        Assert.Contains("EvidenceKind=allocator-owner-internal-runtime-prototype", schema);
        Assert.Contains("ReleaseHookCount", schema);
        Assert.Contains("CallbackStatePinned", schema);
        Assert.Contains("DelegatePinned", schema);
        Assert.Contains("not proof", schema);
        Assert.Contains("SmokeResult=passed", schema);
        Assert.Contains("isRealCallbackRuntimeProof=false", schema);
        Assert.Contains("isRealCallbackRuntimeProof=true", schema);
        Assert.Contains("schema-only", schema);
        Assert.Contains("not-present", schema);

        Assert.Contains("New-RealCallbackTrampolineGateEvidence", readiness);
        Assert.Contains("realCallbackTrampolineGate", readiness);
        Assert.Contains("evidenceKind = \"design-gate-only\"", readiness);
        Assert.Contains("realCallbackRuntimeEvidenceKind = \"not-present\"", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);
        Assert.Contains("real callback trampoline gate:", readiness);
        Assert.Contains("Real callback trampoline missing evidence", readiness);
        Assert.Contains("dry-run/copied-state evidence from real-callback-runtime", readiness);
        Assert.Contains("New-RealCallbackRuntimeEvidenceSchema", readiness);
        Assert.Contains("New-RealCallbackRuntimeEvidence", readiness);
        Assert.Contains("realCallbackRuntimeEvidenceSchema", readiness);
        Assert.Contains("realCallbackRuntimeEvidence", readiness);
        Assert.Contains("callbackRuntimeEvidenceStatus", readiness);
        Assert.Contains("full-package-consumer-report", readiness);
        Assert.Contains("missingSmokeMarkers", readiness);

        Assert.Contains("realCallbackTrampolineGate", readme);
        Assert.Contains("real-callback-trampoline-gate", readme);
        Assert.Contains("realCallbackRuntimeEvidenceSchema", readme);
        Assert.Contains("realCallbackRuntimeEvidence", readme);
        Assert.Contains("EvidenceKind=real-callback-runtime", readme);
        Assert.Contains("design gate only", readme);
        Assert.Contains("not proof that TensorRT callbacks are enabled", readme);
        Assert.Contains("real-callback-runtime", readme);
        Assert.Contains("RealCallbackRuntime=True", packageConsumer);
        Assert.Contains("SmokeResult=passed", packageConsumer);
        Assert.Contains("isRealCallbackRuntimeProof=true", packageConsumer);

        Assert.Contains("\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAllocator\",\"free\",\"IGpuAllocator::free\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAllocator\",\"deallocate\",\"IGpuAllocator::deallocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAsyncAllocator\",\"allocateAsync\",\"IGpuAsyncAllocator::allocateAsync\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAsyncAllocator\",\"deallocateAsync\",\"IGpuAsyncAllocator::deallocateAsync\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void AllocatorOwnerDryRunSkeletonAddsNativeDiagnosticOwnerButKeepsCallbacksDeferred()
    {
        string ownerSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtAllocatorCallbackOwner.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public readonly struct TensorRtAllocatorDryRunRequest", ownerSource);
        Assert.Contains("public readonly struct TensorRtAllocatorDryRunResult", ownerSource);
        Assert.Contains("public delegate TensorRtAllocatorDryRunResult TensorRtAllocatorDryRunHandler", ownerSource);
        Assert.Contains("public sealed class TensorRtAllocatorCallbackOwner", ownerSource);
        Assert.Contains("GCHandle.Alloc(_callbackState)", ownerSource);
        Assert.Contains("keeps managed callback state alive", ownerSource);
        Assert.Contains("public bool IsAttached => false", ownerSource);
        Assert.Contains("public long CallbackInvocationCount", ownerSource);
        Assert.Contains("public long CallbackFailureCount", ownerSource);
        Assert.Contains("public Exception? LastCallbackException", ownerSource);
        Assert.Contains("public string LastDiagnostic", ownerSource);
        Assert.Contains("public TensorRtAllocatorDryRunResult RunDryRunDiagnostic", ownerSource);
        Assert.Contains("public readonly struct TensorRtAllocatorNativeDryRunResult", ownerSource);
        Assert.Contains("public TensorRtAllocatorNativeDryRunResult RunNativeDryRunDiagnostic", ownerSource);
        Assert.Contains("public readonly struct TensorRtAllocatorOwnerStateDryRunResult", ownerSource);
        Assert.Contains("public TensorRtAllocatorOwnerStateDryRunResult RunNativeStateLedgerDryRunDiagnostic", ownerSource);
        Assert.Contains("public readonly struct TensorRtAllocatorCallbackOwnerSnapshot", ownerSource);
        Assert.Contains("public TensorRtAllocatorCallbackOwnerSnapshot RunLifecycleDiagnostic", ownerSource);
        Assert.Contains("public TensorRtAllocatorCallbackOwnerSnapshot GetSnapshot", ownerSource);
        Assert.Contains("RuntimeEvidenceKind => \"not-present\"", ownerSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", ownerSource);
        Assert.Contains("DevicePointerExposed => false", ownerSource);
        Assert.Contains("DevicePointerProduced => false", ownerSource);
        Assert.Contains("BorrowedPointerEscaped => false", ownerSource);
        Assert.Contains("PointerFreeSurfaceReady", ownerSource);
        Assert.Contains("NativeBridgeApi.CreateAllocatorOwnerDryRun(line)", ownerSource);
        Assert.Contains("NativeBridgeApi.EmitAllocatorOwnerDryRunDiagnostic", ownerSource);
        Assert.Contains("NativeBridgeApi.AttachAllocatorOwnerDryRunIntent", ownerSource);
        Assert.Contains("NativeBridgeApi.RecordAllocatorOwnerDryRunAllocationIntent", ownerSource);
        Assert.Contains("NativeBridgeApi.RecordAllocatorOwnerDryRunReleaseIntent", ownerSource);
        Assert.Contains("using SafeTensorRtObjectHandle nativeOwner", ownerSource);
        Assert.Contains("short-lived native diagnostic owner", ownerSource);
        Assert.Contains("does not attach to TensorRT", ownerSource);
        Assert.Contains("does not call", ownerSource);
        Assert.Contains("does not return or own device pointers", ownerSource);
        Assert.Contains("RunInternalSyncAllocatorRuntimePrototype", ownerSource);
        Assert.Contains("GetInternalRuntimePrototypeSnapshot", ownerSource);
        Assert.Contains("TensorRtAllocatorInternalRuntimePrototypeResult", ownerSource);
        Assert.Contains("TensorRtAllocatorInternalRuntimePrototypeCallback", ownerSource);
        Assert.Contains("GCHandle.Alloc(_runtimePrototypeCallback)", ownerSource);
        Assert.Contains("EnterCallback", ownerSource);
        Assert.Contains("ExitCallback", ownerSource);
        Assert.Contains("InFlightCallbackCount", ownerSource);
        Assert.Contains("ReleaseHookCount", ownerSource);
        Assert.Contains("CallbackStatePinned", ownerSource);
        Assert.Contains("DelegatePinned", ownerSource);
        Assert.Contains("RealCallbackRuntime => false", ownerSource);
        Assert.Contains("allocator-owner-internal-runtime-prototype", ownerSource);
        Assert.DoesNotContain("NativeMethods", ownerSource);
        Assert.DoesNotContain("public IntPtr", ownerSource);
        Assert.DoesNotContain("public nint", ownerSource);

        Assert.Contains("allocator-owner-dry-run-diagnostics", smokeProgram);
        Assert.Contains("allocator-owner-native-dry-run-controls", smokeProgram);
        Assert.Contains("allocator-owner-state-ledger-dry-run-controls", smokeProgram);
        Assert.Contains("allocator-owner-internal-runtime-prototype", smokeProgram);
        Assert.Contains("AllocatorOwnerDryRunDiagnostics=", smokeProgram);
        Assert.Contains("AllocatorOwnerNativeDryRunControls=", smokeProgram);
        Assert.Contains("AllocatorOwnerStateLedgerDryRunControls=", smokeProgram);
        Assert.Contains("AllocatorOwnerInternalRuntimePrototype=", smokeProgram);
        Assert.Contains("AllocatorOwnerInternalRuntimePrototypeDispose=", smokeProgram);
        Assert.Contains("AllocatorOwnerInternalRuntimePrototypeException=", smokeProgram);
        Assert.Contains("RealCallbackRuntime", smokeProgram);
        Assert.Contains("RunLifecycleDiagnostic", smokeProgram);
        Assert.Contains("GetSnapshot", smokeProgram);
        Assert.Contains("DevicePointerExposed", smokeProgram);
        Assert.Contains("DevicePointerProduced", smokeProgram);
        Assert.Contains("BorrowedPointerEscaped", smokeProgram);
        Assert.Contains("PointerFreeSurfaceReady", smokeProgram);
        Assert.Contains("new TensorRtAllocatorCallbackOwner", smokeProgram);
        Assert.Contains("RunDryRunDiagnostic(new TensorRtAllocatorDryRunRequest", smokeProgram);
        Assert.Contains("RunNativeDryRunDiagnostic(TensorRtApiLine.TensorRt11", smokeProgram);
        Assert.Contains("RunNativeStateLedgerDryRunDiagnostic(TensorRtApiLine.TensorRt11", smokeProgram);

        Assert.Contains("allocator-owner-dry-run-diagnostics", bridgeConsumer);
        Assert.Contains("allocator-owner-native-dry-run-controls", bridgeConsumer);
        Assert.Contains("allocator-owner-state-ledger-dry-run-controls", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorCallbackOwner", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorDryRunRequest", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorDryRunResult", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorNativeDryRunResult", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorOwnerStateDryRunResult", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorCallbackOwnerSnapshot", bridgeConsumer);
        Assert.Contains("RunDryRunDiagnostic", bridgeConsumer);
        Assert.Contains("RunNativeDryRunDiagnostic", bridgeConsumer);
        Assert.Contains("RunNativeStateLedgerDryRunDiagnostic", bridgeConsumer);
        Assert.Contains("RunLifecycleDiagnostic", bridgeConsumer);
        Assert.Contains("GetSnapshot", bridgeConsumer);

        Assert.Contains("allocator-owner-dry-run-diagnostics", readiness);
        Assert.Contains("allocator-owner-native-dry-run-controls", readiness);
        Assert.Contains("allocator-owner-state-ledger-dry-run-controls", readiness);
        Assert.Contains("hasAllocatorOwnerDryRunDiagnostics", readiness);
        Assert.Contains("hasAllocatorOwnerNativeDryRunControls", readiness);
        Assert.Contains("hasAllocatorOwnerStateLedgerDryRunControls", readiness);
        Assert.Contains("allocatorOwnerInternalRuntimePrototype", readiness);
        Assert.Contains("New-AllocatorOwnerInternalRuntimePrototypeEvidence", readiness);
        Assert.Contains("internal-runtime-prototype", readiness);
        Assert.Contains("TensorRtAllocatorCallbackOwnerSnapshot", readiness);
        Assert.Contains("RunLifecycleDiagnostic", readiness);
        Assert.Contains("DevicePointerExposed", readiness);
        Assert.Contains("PointerFreeSurfaceReady", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAsyncAllocator\",\"allocateAsync\",\"IGpuAsyncAllocator::allocateAsync\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void OutputAllocatorRuntimeGateCopiesDiagnosticsWithoutOpeningCallbacks()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOutputAllocatorRuntimeGate.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-runtime-gate.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("internal sealed class TensorRtOutputAllocatorRuntimeGate", gateSource);
        Assert.Contains("internal readonly struct TensorRtOutputAllocatorRuntimeGateRequest", gateSource);
        Assert.Contains("internal readonly struct TensorRtOutputAllocatorRuntimeGateResult", gateSource);
        Assert.Contains("RunInternalNotifyShapeRuntimeGate", gateSource);
        Assert.Contains("RunInternalReallocateOutputRuntimeGate", gateSource);
        Assert.Contains("GetInternalRuntimeGateSnapshot", gateSource);
        Assert.Contains("TensorRtOutputAllocatorInternalRuntimeGateCallback", gateSource);
        Assert.Contains("GCHandle.Alloc(_runtimeGateCallback)", gateSource);
        Assert.Contains("NotifyShapeCount", gateSource);
        Assert.Contains("ReallocateOutputCount", gateSource);
        Assert.Contains("ShapeRank", gateSource);
        Assert.Contains("ShapeSummary", gateSource);
        Assert.Contains("OutputBufferPointerExposed => false", gateSource);
        Assert.Contains("OutputBufferPointerProduced => false", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("output-buffer-pointer-exposed=false", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("output-allocator-internal-runtime-gate", smokeProgram);
        Assert.Contains("OutputAllocatorInternalRuntimeGateNotifyShape=", smokeProgram);
        Assert.Contains("OutputAllocatorInternalRuntimeGateReallocateOutput=", smokeProgram);
        Assert.Contains("OutputAllocatorInternalRuntimeGateDispose=", smokeProgram);
        Assert.Contains("OutputAllocatorInternalRuntimeGateException=", smokeProgram);
        Assert.Contains("OutputBufferPointerExposed", smokeProgram);
        Assert.Contains("OutputBufferPointerProduced", smokeProgram);
        Assert.Contains("RunInternalNotifyShapeRuntimeGate", smokeProgram);
        Assert.Contains("RunInternalReallocateOutputRuntimeGate", smokeProgram);

        Assert.Contains("outputAllocatorInternalRuntimeGate", readiness);
        Assert.Contains("New-OutputAllocatorInternalRuntimeGateEvidence", readiness);
        Assert.Contains("gate-ready", readiness);
        Assert.Contains("internal-runtime-gate", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("output-allocator-internal-runtime-gate", gateDoc);
        Assert.Contains("RealCallbackRuntime=False", gateDoc);
        Assert.Contains("OutputBufferPointerExposed=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("output-allocator-runtime-gate.md", latest);
        Assert.Contains("output-allocator-runtime-gate.md", docsIndex);
        Assert.Contains("output-allocator-runtime-gate.md", docsToc);
        Assert.Contains("output-allocator-internal-runtime-gate", trampolineGate);
        Assert.Contains("output-allocator-internal-runtime-gate", schema);
        Assert.Contains("outputAllocatorInternalRuntimeGate", runtimeSplitReadme);
        Assert.Contains("output-allocator-internal-runtime-gate", smokeReadme);

        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void SafePublicControlsDocumentThatPointersRemainBorrowed()
    {
        string publicSources = string.Join(
            Environment.NewLine,
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntime.Trt11Controls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilder.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilderConfig.Trt11Diagnostics.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEngine.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtNetworkDefinition.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11Diagnostics.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitter.Trt11Controls.cs"));

        Assert.Contains("不会暴露 recorder 指针或接管其生命周期", publicSources);
        Assert.Contains("TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", publicSources);
        Assert.Contains("不会暴露、持有、增加引用、减少引用或销毁原生 recorder 指针", publicSources);
        Assert.Contains("不会销毁 recorder 或接管其生命周期", publicSources);
        Assert.Contains("不会调用用户 allocator 的 free/deallocate 回调", publicSources);
        Assert.Contains("不会暴露 allocator 指针或接管其生命周期", publicSources);
        Assert.Contains("不会销毁 allocator 或调用用户 reallocate 回调", publicSources);
        Assert.Contains("不会暴露 monitor 指针或接管其生命周期", publicSources);
        Assert.Contains("不会暴露 profiler 指针或接管其生命周期", publicSources);
        Assert.DoesNotContain("public IntPtr", publicSources);
        Assert.DoesNotContain("public nint", publicSources);
    }

    [Fact]
    public void RuntimeErrorRecorderSnapshotUsesCopiedManagedValuesAndKeepsDirectRecorderDeferred()
    {
        string manifest8 = ReadTensorRtManifest("v8", "trt8-runtime-error-recorder-snapshot.manifest.json");
        string manifest10 = ReadTensorRtManifest("v10", "trt10-runtime-error-recorder-snapshot.manifest.json");
        string manifest11 = ReadTensorRtManifest("v11", "trt11-runtime-error-recorder-snapshot.manifest.json");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string types = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string runtimeWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntime.Trt11Controls.cs");
        string snapshotWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtErrorRecorderSnapshot.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeSerializationRefit.cs");
        string deferred8 = ReadTensorRtManifest("v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string deferred10 = ReadTensorRtManifest("v10", "trt10-cross-version-sixth-batch-diagnostics-refitter-deferred.manifest.json");
        string deferred11 = ReadTensorRtManifest("v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11 })
        {
            Assert.Contains("runtime-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("runtime-get-error-recorder-error", manifest);
            Assert.Contains("JYPPX_TensorRtErrorRecorderSnapshotInfo*", manifest);
            Assert.Contains("JYPPX_TensorRtErrorRecordInfo*", manifest);
            Assert.Contains("out NativeTensorRtErrorRecorderSnapshotInfo", manifest);
            Assert.Contains("out NativeTensorRtErrorRecordInfo", manifest);
        }

        Assert.Contains("JYPPX_TensorRtErrorRecorderSnapshotInfo", types);
        Assert.Contains("JYPPX_TensorRtErrorRecordInfo", types);
        Assert.Contains("char description[1024];", types);
        Assert.Contains("jyppx_trt8_runtime_get_error_recorder_snapshot_info", header8);
        Assert.Contains("jyppx_trt10_runtime_get_error_recorder_snapshot_info", header10);
        Assert.Contains("jyppx_trt11_runtime_get_error_recorder_snapshot_info", header11);

        Assert.Contains("public bool TryGetErrorRecorderSnapshot", runtimeWrapper);
        Assert.Contains("TensorRtErrorRecorderSnapshot", snapshotWrapper);
        Assert.Contains("TensorRtErrorRecord", snapshotWrapper);
        Assert.DoesNotContain("public IntPtr", snapshotWrapper + runtimeWrapper);
        Assert.DoesNotContain("public nint", snapshotWrapper + runtimeWrapper);

        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_runtime_get_error_recorder_snapshot_info", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_runtime_get_error_recorder_snapshot_info", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_runtime_get_error_recorder_snapshot_info", interop);
        Assert.Contains("BridgeInfoMapper.ToManaged(error)", interop);

        Assert.Contains("trt8-error-recorder-get-nb-errors-deferred", deferred8);
        Assert.Contains("trt10-error-recorder-get-nb-errors-deferred", deferred10);
        Assert.Contains("trt11-error-recorder-get-nb-errors-deferred", deferred11);
        Assert.Contains("trt11-error-recorder-get-error-desc-deferred", deferred11);
    }

    [Fact]
    public void EngineAndExecutionContextErrorRecorderSnapshotsUseCopiedManagedValues()
    {
        string manifest8 = ReadTensorRtManifest("v8", "trt8-engine-network-context-error-recorder-controls.manifest.json");
        string manifest10 = ReadTensorRtManifest("v10", "trt10-engine-network-context-error-recorder-controls.manifest.json");
        string manifest11 = ReadTensorRtManifest("v11", "trt11-thirteenth-batch.manifest.json");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string commonBoundary = ReadSource("native", "src", "tensorrt", "common", "error_recorder_boundary_controls.inc");
        string trt11Boundary = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs");
        string engineWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEngine.Trt11BoundaryControls.cs");
        string contextWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11BoundaryControls.cs");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11 })
        {
            Assert.Contains("engine-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("engine-get-error-recorder-error", manifest);
            Assert.Contains("execution-context-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("execution-context-get-error-recorder-error", manifest);
            Assert.Contains("JYPPX_TensorRtErrorRecorderSnapshotInfo*", manifest);
            Assert.Contains("JYPPX_TensorRtErrorRecordInfo*", manifest);
            Assert.Contains("out NativeTensorRtErrorRecorderSnapshotInfo", manifest);
            Assert.Contains("out NativeTensorRtErrorRecordInfo", manifest);
        }

        foreach (string header in new[] { header8, header10, header11 })
        {
            Assert.Contains("engine_get_error_recorder_snapshot_info", header);
            Assert.Contains("engine_get_error_recorder_error", header);
            Assert.Contains("execution_context_get_error_recorder_snapshot_info", header);
            Assert.Contains("execution_context_get_error_recorder_error", header);
        }

        Assert.Contains("engine_get_error_recorder_snapshot_info_with_seh_guard", commonBoundary);
        Assert.Contains("engine_get_error_recorder_error_with_seh_guard", commonBoundary);
        Assert.Contains("execution_context_get_error_recorder_snapshot_info_with_seh_guard", commonBoundary);
        Assert.Contains("execution_context_get_error_recorder_error_with_seh_guard", commonBoundary);
        Assert.Contains("error_recorder_boundary_reset_error_recorder_snapshot_info", commonBoundary);
        Assert.Contains("error_recorder_boundary_reset_error_record_info", commonBoundary);
        Assert.Contains("copy_c_string(recorder->getErrorDesc(index), out_error->description", commonBoundary);

        Assert.Contains("trt11_engine_get_error_recorder_snapshot_info_with_seh_guard", trt11Boundary);
        Assert.Contains("trt11_engine_get_error_recorder_error_with_seh_guard", trt11Boundary);
        Assert.Contains("trt11_execution_context_get_error_recorder_snapshot_info_with_seh_guard", trt11Boundary);
        Assert.Contains("trt11_execution_context_get_error_recorder_error_with_seh_guard", trt11Boundary);
        Assert.Contains("trt11_reset_runtime_error_recorder_snapshot_info", trt11Boundary);
        Assert.Contains("trt11_reset_runtime_error_record_info", trt11Boundary);

        Assert.Contains("GetEngineErrorRecorderSnapshot", interop);
        Assert.Contains("GetExecutionContextErrorRecorderSnapshot", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_engine_get_error_recorder_snapshot_info", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_engine_get_error_recorder_error", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_execution_context_get_error_recorder_snapshot_info", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_execution_context_get_error_recorder_error", interop);
        Assert.Contains("BridgeInfoMapper.ToManaged(error)", interop);
        Assert.Contains("BridgeInfoMapper.ToManaged(line, info", interop);

        Assert.Contains("public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", engineWrapper);
        Assert.Contains("public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", contextWrapper);
        Assert.DoesNotContain("public IntPtr", engineWrapper + contextWrapper);
        Assert.DoesNotContain("public nint", engineWrapper + contextWrapper);
    }

    [Fact]
    public void BuilderNetworkAndInspectorErrorRecorderSnapshotsUseCopiedManagedValues()
    {
        string manifest8 = ReadTensorRtManifest("v8", "trt8-owner-error-recorder-snapshot.manifest.json");
        string manifest10 = ReadTensorRtManifest("v10", "trt10-owner-error-recorder-snapshot.manifest.json");
        string manifest11 = ReadTensorRtManifest("v11", "trt11-owner-error-recorder-snapshot.manifest.json");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string commonBoundary = ReadSource("native", "src", "tensorrt", "common", "error_recorder_boundary_controls.inc");
        string trt11Boundary = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs");
        string builderWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilder.Trt11BoundaryControls.cs");
        string networkWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtNetworkDefinition.Trt11BoundaryControls.cs");
        string inspectorWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEngineInspector.Trt11Diagnostics.cs");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11 })
        {
            Assert.Contains("builder-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("builder-get-error-recorder-error", manifest);
            Assert.Contains("network-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("network-get-error-recorder-error", manifest);
            Assert.Contains("engine-inspector-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("engine-inspector-get-error-recorder-error", manifest);
            Assert.Contains("JYPPX_TensorRtErrorRecorderSnapshotInfo*", manifest);
            Assert.Contains("JYPPX_TensorRtErrorRecordInfo*", manifest);
        }

        foreach (string header in new[] { header8, header10, header11 })
        {
            Assert.Contains("builder_get_error_recorder_snapshot_info", header);
            Assert.Contains("builder_get_error_recorder_error", header);
            Assert.Contains("network_get_error_recorder_snapshot_info", header);
            Assert.Contains("network_get_error_recorder_error", header);
            Assert.Contains("engine_inspector_get_error_recorder_snapshot_info", header);
            Assert.Contains("engine_inspector_get_error_recorder_error", header);
        }

        Assert.Contains("get_builder_payload_for_error_recorder_boundary", commonBoundary);
        Assert.Contains("owner_get_error_recorder_snapshot_info_with_seh_guard", commonBoundary);
        Assert.Contains("owner_get_error_recorder_error_with_seh_guard", commonBoundary);
        Assert.Contains("trt11_owner_get_error_recorder_snapshot_info_with_seh_guard", trt11Boundary);
        Assert.Contains("trt11_owner_get_error_recorder_error_with_seh_guard", trt11Boundary);
        Assert.Contains("GetBuilderErrorRecorderSnapshot", interop);
        Assert.Contains("GetNetworkErrorRecorderSnapshot", interop);
        Assert.Contains("GetEngineInspectorErrorRecorderSnapshot", interop);
        Assert.Contains("jyppx_trt8_builder_get_error_recorder_error", interop);
        Assert.Contains("jyppx_trt10_network_get_error_recorder_snapshot_info", interop);
        Assert.Contains("jyppx_trt11_engine_inspector_get_error_recorder_error", interop);
        Assert.Contains("TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", builderWrapper);
        Assert.Contains("TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", networkWrapper);
        Assert.Contains("TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", inspectorWrapper);
        Assert.Contains("builder.TryGetErrorRecorderSnapshot", smoke);
        Assert.Contains("network.TryGetErrorRecorderSnapshot", smoke);
        Assert.Contains("inspector.TryGetErrorRecorderSnapshot", smoke);
        Assert.DoesNotContain("public IntPtr", builderWrapper + networkWrapper + inspectorWrapper);
        Assert.DoesNotContain("public nint", builderWrapper + networkWrapper + inspectorWrapper);
    }

    [Fact]
    public void ErrorRecorderSnapshotsCopyInterfaceInfoWithoutExposingNativeRecorder()
    {
        string nativeTypes = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string nativeBoundary = ReadSource("native", "src", "tensorrt", "common", "error_recorder_boundary_controls.inc");
        string runtimeControls = ReadSource("native", "src", "tensorrt", "common", "runtime_controls.inc");
        string refitterControls = ReadSource("native", "src", "tensorrt", "common", "refitter_controls.inc");
        string trt11RuntimeSerializationRefit = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "runtime_serialization_refit.inc");
        string trt11BoundaryControls = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");
        string managedStructs = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeStructs.cs");
        string mapper = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "BridgeInfoMapper.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtErrorRecorderSnapshot.cs");
        string boundaryInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs");
        string runtimeRefitInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeSerializationRefit.cs");

        Assert.Contains("interface_info_available", nativeTypes);
        Assert.Contains("interface_info_kind[128]", nativeTypes);
        Assert.Contains("interface_info_major", nativeTypes);
        Assert.Contains("interface_info_minor", nativeTypes);

        Assert.Contains("error_recorder_boundary_copy_interface_info", nativeBoundary);
        Assert.Contains("recorder->getInterfaceInfo()", nativeBoundary);
        Assert.Contains("#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10", nativeBoundary);
        Assert.Contains("copy_c_string(interface_info.kind, out_info->interface_info_kind", nativeBoundary);
        Assert.Contains("out_info->interface_info_available = JYPPX_FALSE;", nativeBoundary);
        Assert.Contains("error_recorder_boundary_copy_interface_info(recorder, out_info);", nativeBoundary);

        Assert.Contains("runtime_copy_error_recorder_interface_info", runtimeControls);
        Assert.Contains("runtime_copy_error_recorder_interface_info(recorder, out_info);", runtimeControls);
        Assert.Contains("refitter_copy_error_recorder_interface_info", refitterControls);
        Assert.Contains("refitter_copy_error_recorder_interface_info(recorder, out_info);", refitterControls);

        Assert.Contains("trt11_copy_runtime_error_recorder_interface_info", trt11RuntimeSerializationRefit);
        Assert.Contains("out_info->interface_info_available = JYPPX_FALSE;", trt11RuntimeSerializationRefit);
        Assert.Contains("out_info->interface_info_kind[0] = '\\0';", trt11RuntimeSerializationRefit);
        Assert.Contains("trt11_copy_runtime_error_recorder_interface_info(recorder, out_info);", trt11RuntimeSerializationRefit);

        Assert.Contains("trt11_reset_error_recorder_interface_info", trt11BoundaryControls);
        Assert.Contains("trt11_copy_error_recorder_interface_info", trt11BoundaryControls);
        Assert.Contains("trt11_copy_error_recorder_interface_info(recorder, out_info);", trt11BoundaryControls);
        Assert.True(
            System.Text.RegularExpressions.Regex.Matches(
                trt11BoundaryControls,
                System.Text.RegularExpressions.Regex.Escape("trt11_copy_error_recorder_interface_info(recorder, out_info);")).Count >= 4);
        Assert.True(
            System.Text.RegularExpressions.Regex.Matches(
                trt11BoundaryControls,
                System.Text.RegularExpressions.Regex.Escape("trt11_reset_error_recorder_interface_info(out_info);")).Count >= 6);

        Assert.Contains("public int InterfaceInfoAvailable;", managedStructs);
        Assert.Contains("public int InterfaceInfoMajor;", managedStructs);
        Assert.Contains("public int InterfaceInfoMinor;", managedStructs);
        Assert.Contains("[MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]", managedStructs);
        Assert.Contains("public byte[] InterfaceInfoKind;", managedStructs);

        Assert.Contains("public static TensorRtErrorRecorderSnapshot ToManaged(", mapper);
        Assert.Contains("ReadFixedUtf8(value.InterfaceInfoKind)", mapper);
        Assert.Contains("value.InterfaceInfoAvailable != 0", mapper);
        Assert.Contains("records ?? Array.Empty<TensorRtErrorRecord>()", mapper);

        Assert.Contains("public bool InterfaceInfoAvailable", snapshot);
        Assert.Contains("public TensorRtInterfaceInfo InterfaceInfo", snapshot);
        Assert.Contains("InterfaceInfoAvailable ? InterfaceInfo.ToString() : \"n/a\"", snapshot);
        Assert.DoesNotContain("public IntPtr", snapshot);
        Assert.DoesNotContain("public nint", snapshot);

        Assert.Contains("BridgeInfoMapper.ToManaged(line, info, Array.Empty<TensorRtErrorRecord>())", boundaryInterop);
        Assert.Contains("BridgeInfoMapper.ToManaged(line, info, records)", boundaryInterop);
        Assert.Contains("BridgeInfoMapper.ToManaged(line, info, Array.Empty<TensorRtErrorRecord>())", runtimeRefitInterop);
        Assert.Contains("BridgeInfoMapper.ToManaged(line, info, records)", runtimeRefitInterop);
    }

    [Fact]
    public void NativeSafeControlsUseBooleanProbeOrNullClearPattern()
    {
        string runtimeControls = ReadSource("native", "src", "tensorrt", "common", "runtime_controls.inc");
        string allocatorControls = ReadSource("native", "src", "tensorrt", "common", "execution_context_allocator_controls.inc");
        string refitterControls = ReadSource("native", "src", "tensorrt", "common", "refitter_controls.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Diagnostics = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "diagnostics.inc");
        string trt11BoundaryControls = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");

        Assert.Contains("runtime_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", runtimeControls);
        Assert.Contains("runtime_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", runtimeControls);
        Assert.Contains("runtime_payload->setErrorRecorder(nullptr);", runtimeControls);
        Assert.Contains("runtime_payload->setGpuAllocator(nullptr);", runtimeControls);
        Assert.Contains("runtime_has_error_recorder_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_has_logger_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_get_error_recorder_snapshot_info_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_get_error_recorder_error_with_seh_guard", runtimeControls);
        Assert.Contains("recorder->getNbErrors()", runtimeControls);
        Assert.Contains("copy_c_string(recorder->getErrorDesc(index), out_error->description", runtimeControls);
        Assert.Contains("runtime_clear_error_recorder_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_clear_gpu_allocator_with_seh_guard", runtimeControls);

        Assert.Contains("execution_context_has_output_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("execution_context_has_temporary_storage_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("execution_context_clear_output_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("execution_context_clear_temporary_storage_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE", allocatorControls);
        Assert.Contains("context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", allocatorControls);
        Assert.Contains("context_payload->setOutputAllocator(tensor_name, nullptr)", allocatorControls);
        Assert.Contains("context_payload->setTemporaryStorageAllocator(nullptr)", allocatorControls);
        Assert.Contains("#include \"../common/execution_context_allocator_controls.inc\"", trt8Api);
        Assert.Contains("#include \"../common/execution_context_allocator_controls.inc\"", trt10Api);

        Assert.Contains("payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", refitterControls);
        Assert.Contains("payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", refitterControls);
        Assert.Contains("payload->setErrorRecorder(nullptr);", refitterControls);
        Assert.Contains("refitter_has_logger_with_seh_guard", refitterControls);
        Assert.Contains("refitter_has_error_recorder_with_seh_guard", refitterControls);
        Assert.Contains("refitter_clear_error_recorder_with_seh_guard", refitterControls);

        Assert.Contains("config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11Diagnostics);
        Assert.Contains("config_payload->setProgressMonitor(nullptr);", trt11Diagnostics);
        Assert.Contains("context_payload->setOutputAllocator(tensor_name, nullptr)", trt11Diagnostics);
        Assert.Contains("context_payload->setTemporaryStorageAllocator(nullptr)", trt11Diagnostics);
        Assert.Contains("context_payload->setDebugListener(nullptr)", trt11Diagnostics);
        Assert.Contains("context_payload->getProfiler() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11Diagnostics);
        Assert.Contains("trt11_builder_config_has_progress_monitor_with_seh_guard", trt11Diagnostics);
        Assert.Contains("trt11_execution_context_clear_output_allocator_with_seh_guard", trt11Diagnostics);
        Assert.Contains("trt11_execution_context_clear_temporary_storage_allocator_with_seh_guard", trt11Diagnostics);
        Assert.Contains("trt11_execution_context_clear_debug_listener_with_seh_guard", trt11Diagnostics);
        Assert.Contains("trt11_execution_context_has_profiler_with_seh_guard", trt11Diagnostics);

        Assert.Contains("trt11_builder_clear_gpu_allocator_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_builder_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_builder_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_engine_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_engine_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_execution_context_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_execution_context_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_network_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_network_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("builder_payload->setGpuAllocator(nullptr);", trt11BoundaryControls);
        Assert.Contains("builder_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11BoundaryControls);
        Assert.Contains("builder_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11BoundaryControls);
        Assert.Contains("builder_payload->setErrorRecorder(nullptr);", trt11BoundaryControls);
        Assert.Contains("engine_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11BoundaryControls);
        Assert.Contains("engine_payload->setErrorRecorder(nullptr);", trt11BoundaryControls);
        Assert.Contains("context_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11BoundaryControls);
        Assert.Contains("context_payload->setErrorRecorder(nullptr);", trt11BoundaryControls);
        Assert.Contains("network_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", trt11BoundaryControls);
        Assert.Contains("network_payload->setErrorRecorder(nullptr);", trt11BoundaryControls);
    }

    [Fact]
    public void BuilderAndRuntimeLoggerPresenceAreSafeBooleanProbes()
    {
        string manifest8Builder = ReadTensorRtManifest("v8", "trt8-builder-callback-boundary-controls.manifest.json");
        string manifest10Builder = ReadTensorRtManifest("v10", "trt10-builder-and-debug-listener-safe-controls.manifest.json");
        string manifest11Builder = ReadTensorRtManifest("v11", "trt11-thirteenth-batch.manifest.json");
        string manifest8Runtime = ReadTensorRtManifest("v8", "trt8-thirtieth-batch-runtime-controls.manifest.json");
        string manifest10Runtime = ReadTensorRtManifest("v10", "trt10-thirtieth-batch-runtime-controls.manifest.json");
        string manifest11Runtime = ReadTensorRtManifest("v11", "trt11-twelfth-batch.manifest.json");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string builderControls = ReadSource("native", "src", "tensorrt", "common", "builder_callback_boundary_controls.inc");
        string runtimeControls = ReadSource("native", "src", "tensorrt", "common", "runtime_controls.inc");
        string trt11BoundaryControls = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");
        string trt11RuntimeRefit = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "runtime_serialization_refit.inc");
        string builderInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs");
        string runtimeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeSerializationRefit.cs");
        string builderWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilder.Trt11BoundaryControls.cs");
        string runtimeWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntime.Trt11Controls.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string trt8Deferred = ReadTensorRtManifest("v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string trt10Deferred = ReadTensorRtManifest("v10", "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");
        string trt11Deferred = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        foreach (string manifest in new[] { manifest8Builder, manifest10Builder, manifest11Builder })
        {
            Assert.Contains("builder-has-logger", manifest);
            Assert.Contains("out_has_logger", manifest);
            Assert.Contains("\"type\": \"JYPPX_Boolean*\"", manifest);
        }

        foreach (string manifest in new[] { manifest8Runtime, manifest10Runtime, manifest11Runtime })
        {
            Assert.Contains("runtime-has-logger", manifest);
            Assert.Contains("out_has_logger", manifest);
            Assert.Contains("\"type\": \"JYPPX_Boolean*\"", manifest);
        }

        Assert.Contains("jyppx_trt8_builder_has_logger", header8);
        Assert.Contains("jyppx_trt10_builder_has_logger", header10);
        Assert.Contains("jyppx_trt11_builder_has_logger", header11);
        Assert.Contains("jyppx_trt8_runtime_has_logger", header8);
        Assert.Contains("jyppx_trt10_runtime_has_logger", header10);
        Assert.Contains("jyppx_trt11_runtime_has_logger", header11);

        Assert.Contains("builder_has_logger_with_seh_guard", builderControls);
        Assert.Contains("builder_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", builderControls);
        Assert.Contains("runtime_has_logger_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", runtimeControls);
        Assert.Contains("trt11_builder_has_logger_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_runtime_has_logger_with_seh_guard", trt11RuntimeRefit);

        Assert.Contains("public static bool HasBuilderLogger", builderInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_builder_has_logger", builderInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_builder_has_logger", builderInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_builder_has_logger", builderInterop);
        Assert.Contains("public static bool HasRuntimeLogger", runtimeInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_runtime_has_logger", runtimeInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_runtime_has_logger", runtimeInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_runtime_has_logger", runtimeInterop);

        Assert.Contains("public bool HasLogger => NativeBridgeApi.HasBuilderLogger", builderWrapper);
        Assert.Contains("public bool HasLogger => NativeBridgeApi.HasRuntimeLogger", runtimeWrapper);
        Assert.Contains("不会暴露 logger 指针或接管其生命周期", builderWrapper + runtimeWrapper);
        Assert.DoesNotContain("public IntPtr", builderWrapper + runtimeWrapper);
        Assert.DoesNotContain("public nint", builderWrapper + runtimeWrapper);

        Assert.Contains("runtime.HasLogger", smokeProgram);
        Assert.Contains("builder.HasLogger", smokeProgram);
        Assert.Contains("Logger={hasLogger}", smokeProgram);

        Assert.Contains("trt8-runtime-get-logger-deferred", trt8Deferred);
        Assert.Contains("trt10-runtime-get-logger-deferred", trt10Deferred);
        Assert.Contains("trt11-runtime-get-logger-deferred", trt11Deferred);
        Assert.Contains("trt11-builder-get-logger-deferred", trt11Deferred);
    }

    [Fact]
    public void CallbackAllocatorInterfaceInfoUsesCopiedMetadataOnly()
    {
        string manifest10 = ReadTensorRtManifest("v10", "trt10-callback-interface-info.manifest.json");
        string manifest11 = ReadTensorRtManifest("v11", "trt11-callback-interface-info.manifest.json");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "execution_context_callback_interface_info.inc");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.CallbackInterfaceInfo.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");

        foreach (string manifest in new[] { manifest10, manifest11 })
        {
            Assert.Contains("execution-context-output-allocator-get-interface-info", manifest);
            Assert.Contains("execution-context-temporary-storage-allocator-get-interface-info", manifest);
            Assert.Contains("execution-context-debug-listener-get-interface-info", manifest);
            Assert.Contains("\"type\": \"char*\"", manifest);
            Assert.Contains("\"managedType\": \"byte[]\"", manifest);
            Assert.Contains("\"name\": \"out_major\"", manifest);
            Assert.Contains("\"name\": \"out_minor\"", manifest);
        }

        Assert.Contains("jyppx_trt10_execution_context_get_output_allocator_interface_info", header10);
        Assert.Contains("jyppx_trt10_execution_context_get_temporary_storage_allocator_interface_info", header10);
        Assert.Contains("jyppx_trt10_execution_context_get_debug_listener_interface_info", header10);
        Assert.Contains("jyppx_trt11_execution_context_get_output_allocator_interface_info", header11);
        Assert.Contains("jyppx_trt11_execution_context_get_temporary_storage_allocator_interface_info", header11);
        Assert.Contains("jyppx_trt11_execution_context_get_debug_listener_interface_info", header11);

        Assert.Contains("execution_context_copy_borrowed_interface_info", nativeSource);
        Assert.Contains("context_payload->getOutputAllocator(tensor_name)", nativeSource);
        Assert.Contains("context_payload->getTemporaryStorageAllocator()", nativeSource);
        Assert.Contains("context_payload->getDebugListener()", nativeSource);
        Assert.Contains("JYPPX_STATUS_NOT_FOUND", nativeSource);
        Assert.Contains("copy_interface_info_to_buffer(info, output_buffer, output_buffer_size, out_required_size, out_major, out_minor)", nativeSource);
        Assert.DoesNotContain("reinterpret_cast<uintptr_t>", nativeSource);

        Assert.Contains("#include \"../common/execution_context_callback_interface_info.inc\"", trt10Api);
        Assert.Contains("#include \"../common/execution_context_callback_interface_info.inc\"", trt11Api);

        Assert.Contains("GetExecutionContextOutputAllocatorInterfaceInfo", interop);
        Assert.Contains("GetExecutionContextTemporaryStorageAllocatorInterfaceInfo", interop);
        Assert.Contains("GetExecutionContextDebugListenerInterfaceInfo", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_execution_context_get_output_allocator_interface_info", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_execution_context_get_debug_listener_interface_info", interop);

        Assert.Contains("public bool TryGetOutputAllocatorInterfaceInfo", wrapper);
        Assert.Contains("public bool TryGetTemporaryStorageAllocatorInterfaceInfo", wrapper);
        Assert.Contains("public bool TryGetDebugListenerInterfaceInfo", wrapper);
        Assert.Contains("不会暴露、保留或接管 borrowed allocator 指针", wrapper);
        Assert.Contains("不会暴露、保留或接管 borrowed debug-listener 指针", wrapper);
        Assert.DoesNotContain("public IntPtr", wrapper);
        Assert.DoesNotContain("public nint", wrapper);

        Assert.Contains("context.TryGetOutputAllocatorInterfaceInfo(outputTensorName", smokeProgram);
        Assert.Contains("context.TryGetTemporaryStorageAllocatorInterfaceInfo", smokeProgram);
        Assert.Contains("context.TryGetDebugListenerInterfaceInfo", smokeProgram);
    }

    [Fact]
    public void NativeDeferredMessagesNameCallbackOwnershipRisk()
    {
        string trt11Deferred = ReadSource("native", "src", "tensorrt", "v11", "modules", "deferred", "twenty_third_batch_deferred.inc");
        string trt10DiagnosticsDeferred = ReadSource("native", "src", "tensorrt", "v10", "modules", "deferred", "cross_version_diagnostics_refitter_deferred.inc");

        Assert.Contains("borrowed logger callback pointer", trt11Deferred);
        Assert.Contains("managed allocator contract", trt11Deferred);
        Assert.Contains("not safe as standalone bridge calls", trt11Deferred);
        Assert.Contains("managed callback ownership", trt11Deferred);

        Assert.Contains("application-owned recorder callback lifetime", trt10DiagnosticsDeferred);
        Assert.Contains("borrowed logger discovery and logger lifetime", trt10DiagnosticsDeferred);
        Assert.Contains("application-owned managed callback bridge", trt10DiagnosticsDeferred);
    }

    [Fact]
    public void CoverageMatrixPromotesManagedCallbackOwnerEvidenceWithoutOpeningAllocatorCallbacks()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("\"IProfiler\",\"reportLayerTime\",\"IProfiler::reportLayerTime\",\"diagnostics\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("profiler-create-with-callback", comparison);
        Assert.Contains("profiler-emit-diagnostic", comparison);
        Assert.Contains("profiler-report-layer-time-deferred", comparison);

        Assert.Contains("\"IProgressMonitor\",\"phaseStart\",\"IProgressMonitor::phaseStart\",\"other\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IProgressMonitor\",\"stepComplete\",\"IProgressMonitor::stepComplete\",\"other\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IProgressMonitor\",\"phaseFinish\",\"IProgressMonitor::phaseFinish\",\"other\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("progress-monitor-create-with-callback", comparison);
        Assert.Contains("progress-monitor-emit-diagnostic", comparison);
        Assert.Contains("progress-monitor-step-complete-deferred", comparison);

        Assert.Contains("\"IBuilder\",\"getLogger\",\"IBuilder::getLogger\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IRuntime\",\"getLogger\",\"IRuntime::getLogger\",\"runtime-serialization\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("builder-has-logger", comparison);
        Assert.Contains("runtime-has-logger", comparison);
        Assert.Contains("builder-get-logger-deferred", comparison);
        Assert.Contains("runtime-get-logger-deferred", comparison);

        Assert.Contains("\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("gpu-allocator-allocate-deferred", comparison);
        Assert.Contains("output-allocator-reallocate-output-deferred", comparison);
    }

    [Fact]
    public void CallbackAllocatorSafeControlsSmokeProvidesSkippableConsumerCoverage()
    {
        string project = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "CallbackAllocatorSafeControlsSmokeRunner.csproj");
        string program = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");

        Assert.Contains("CallbackAllocatorSafeControlsSmokeRunner", smokeReadme);
        Assert.Contains("allocator-owner-internal-runtime-prototype", smokeReadme);
        Assert.Contains("output-allocator-internal-runtime-gate", smokeReadme);
        Assert.Contains("RealCallbackRuntime=False", smokeReadme);
        Assert.Contains("not proof", smokeReadme);
        Assert.Contains("CallbackAllocatorSafeControlsSmokeRunner.csproj", solution);
        Assert.Contains("<ProjectReference Include=\"..\\..\\src\\JYPPX.TensorRtSharp\\JYPPX.TensorRtSharp.csproj\" />", project);

        Assert.Contains("--dependency-probe-only", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
        Assert.Contains("SafeControlSurface=allocator-debug-listener-safe-controls;callback-interface-info-safe-controls", program);
        Assert.Contains("allocator-owner-dry-run-diagnostics", program);
        Assert.Contains("allocator-owner-internal-runtime-prototype", program);
        Assert.Contains("output-allocator-internal-runtime-gate", program);
        Assert.Contains("AllocatorOwnerDryRunDiagnostics=", program);
        Assert.Contains("AllocatorOwnerInternalRuntimePrototype=", program);
        Assert.Contains("AllocatorOwnerInternalRuntimePrototypeDispose=", program);
        Assert.Contains("AllocatorOwnerInternalRuntimePrototypeException=", program);
        Assert.Contains("OutputAllocatorInternalRuntimeGateNotifyShape=", program);
        Assert.Contains("OutputAllocatorInternalRuntimeGateReallocateOutput=", program);
        Assert.Contains("OutputAllocatorInternalRuntimeGateDispose=", program);
        Assert.Contains("OutputAllocatorInternalRuntimeGateException=", program);
        Assert.Contains("RealCallbackRuntime", program);
        Assert.Contains("OutputBufferPointerExposed", program);
        Assert.Contains("OutputBufferPointerProduced", program);
        Assert.Contains("InFlightCallbackCount", program);
        Assert.Contains("ReleaseHookCount", program);
        Assert.Contains("CallbackStatePinned", program);
        Assert.Contains("DelegatePinned", program);
        Assert.Contains("CallbackInterfaceInfoSafeControls=TryGetOutputAllocatorInterfaceInfo;TryGetTemporaryStorageAllocatorInterfaceInfo;TryGetDebugListenerInterfaceInfo", program);
        Assert.Contains("CallbackAllocatorSafeControlSummary=GetCallbackAllocatorSafeControlSummary;TensorRtExecutionContextCallbackAllocatorSafeControlSummary;copied-metadata-only;pointer-free;not-runtime-proof", program);
        Assert.Contains("context.GetCallbackAllocatorSafeControlSummary(outputTensorName)", program);
        Assert.Contains("FormatCallbackAllocatorSafeControlSummary", program);
        Assert.Contains("execution-context-callback-allocator-safe-control-summary", program);
        Assert.Contains("Skipped=True Reason=AdapterNotReady", program);
        Assert.Contains("FullEngineContextCallbackAllocatorSafeControlsRequireTensorRt11", program);
        Assert.Contains("ProgressMonitor=Skipped/RequiresTensorRt10Or11", program);
        Assert.Contains("BridgeStatusCode.RuntimeError", program);
        Assert.Contains("structured exception", program);

        Assert.Contains("runtime.ClearErrorRecorder();", program);
        Assert.Contains("runtime.GetDiagnosticSnapshot()", program);
        string runtimeDiagnosticSnapshotSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntimeDiagnosticSnapshot.cs");
        string refitterDiagnosticSnapshotSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitterDiagnosticSnapshot.cs");
        Assert.Contains("public sealed class TensorRtRuntimeDiagnosticSnapshot", runtimeDiagnosticSnapshotSource);
        Assert.Contains("public TensorRtRuntimeDiagnosticSummary ToSummary()", runtimeDiagnosticSnapshotSource);
        Assert.Contains("public sealed class TensorRtRuntimeDiagnosticSummary", runtimeDiagnosticSnapshotSource);
        Assert.Contains("This method only reads managed snapshot values. It does not call TensorRT, expose native pointers, or promote runtime proof.", runtimeDiagnosticSnapshotSource);
        Assert.Contains("public int CopiedErrorRecordCount { get; }", runtimeDiagnosticSnapshotSource);
        Assert.Contains("public int DiagnosticCount { get; }", runtimeDiagnosticSnapshotSource);
        Assert.DoesNotContain("public IntPtr", runtimeDiagnosticSnapshotSource);
        Assert.DoesNotContain("public nint", runtimeDiagnosticSnapshotSource);
        Assert.Contains("public TensorRtRefitterDiagnosticSummary ToSummary()", refitterDiagnosticSnapshotSource);
        Assert.Contains("public sealed class TensorRtRefitterDiagnosticSummary", refitterDiagnosticSnapshotSource);
        Assert.Contains("This method only reads managed snapshot values. It does not call TensorRT, expose native pointers, or promote runtime proof.", refitterDiagnosticSnapshotSource);
        Assert.Contains("public int CopiedMissingNamedWeightCount { get; }", refitterDiagnosticSnapshotSource);
        Assert.Contains("public int CopiedAllNamedWeightCount { get; }", refitterDiagnosticSnapshotSource);
        Assert.DoesNotContain("public IntPtr", refitterDiagnosticSnapshotSource);
        Assert.DoesNotContain("public nint", refitterDiagnosticSnapshotSource);
        Assert.Contains("RuntimeDiagnosticSnapshot=GetDiagnosticSnapshot;TensorRtRuntimeDiagnosticSnapshot;pointer-free", program);
        Assert.Contains("RuntimeDiagnosticSummary=ToSummary;TensorRtRuntimeDiagnosticSummary;pointer-free;not-runtime-proof", program);
        Assert.Contains("RefitterDiagnosticSummary=ToSummary;TensorRtRefitterDiagnosticSummary;copied-inventory;pointer-free;not-runtime-proof", program);
        Assert.Contains("RuntimeDiagnosticSnapshot={diagnosticSnapshot.HasLogger}/{diagnosticSnapshot.HasErrorRecorder}/{diagnosticSnapshot.ErrorRecorder.ErrorCount}/{diagnosticSnapshot.Diagnostics.Count}", program);
        Assert.Contains("RuntimeDiagnosticSummary={diagnosticSummary.HasLogger}/{diagnosticSummary.HasErrorRecorder}/{diagnosticSummary.ErrorCount}/{diagnosticSummary.CopiedErrorRecordCount}/{diagnosticSummary.DiagnosticCount}", program);
        Assert.Contains("runtime.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", program);
        Assert.Contains("Snapshot={snapshotAvailable}/{snapshot.ErrorCount}/{snapshot.Records.Count}/Overflow={snapshot.HasOverflowed}", program);
        Assert.Contains("runtime.ClearGpuAllocator();", program);
        Assert.Contains("builder.ClearErrorRecorder();", program);
        Assert.Contains("builder.ClearGpuAllocator();", program);
        Assert.Contains("ProbeBuilderConfig(line, config)", program);
        Assert.Contains("config.ClearProgressMonitor();", program);
        Assert.Contains("network.ClearErrorRecorder();", program);
        Assert.Contains("engine.ClearErrorRecorder();", program);
        Assert.Contains("inspector.ClearErrorRecorder();", program);
        Assert.Contains("context.ClearErrorRecorder();", program);
        Assert.Contains("context.HasOutputAllocator(outputTensorName)", program);
        Assert.Contains("context.ClearOutputAllocator(outputTensorName)", program);
        Assert.Contains("context.HasTemporaryStorageAllocator", program);
        Assert.Contains("context.ClearTemporaryStorageAllocator()", program);
        Assert.Contains("context.HasDebugListener", program);
        Assert.Contains("context.ClearDebugListener()", program);
        Assert.Contains("context.HasNativeProfiler", program);
        Assert.Contains("context.HasProfiler", program);
        Assert.Contains("context.ClearProfiler();", program);
        Assert.Contains("ProfilerNative=", program);
    }

    [Fact]
    public void ExecutionContextCallbackStateSnapshotUsesCopiedStructAndLeavesCallbacksDeferred()
    {
        string manifest8 = ReadTensorRtManifest("v8", "trt8-execution-context-allocator-safe-controls.manifest.json");
        string manifest10 = ReadTensorRtManifest("v10", "trt10-execution-context-allocator-safe-controls.manifest.json");
        string manifest11 = ReadTensorRtManifest("v11", "trt11-execution-context-callback-state-snapshot.manifest.json");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string types = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "execution_context_callback_state_snapshot.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.ExecutionContextCallbackState.cs");
        string nativeStructs = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeStructs.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string snapshotWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContextCallbackStateSnapshot.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string trt8Deferred = ReadTensorRtManifest("v8", "trt8-cross-version-tenth-batch-other-deferred.manifest.json");
        string trt10Deferred = ReadTensorRtManifest("v10", "trt10-cross-version-third-batch-other-deferred.manifest.json");
        string trt11Deferred = ReadTensorRtManifest("v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11 })
        {
            Assert.Contains("execution-context-get-callback-state-snapshot", manifest);
            Assert.Contains("execution-context-clear-callback-state", manifest);
            Assert.Contains("JYPPX_TensorRtExecutionContextCallbackStateInfo*", manifest);
            Assert.Contains("\"direction\": \"out\"", manifest);
        }

        Assert.Contains("JYPPX_TensorRtExecutionContextCallbackStateInfo", types);
        Assert.Contains("has_output_allocator", types);
        Assert.Contains("has_temporary_storage_allocator", types);
        Assert.Contains("has_debug_listener", types);
        Assert.Contains("output_allocator_interface_info_available", types);
        Assert.Contains("debug_listener_clear_supported", types);
        Assert.Contains("char last_diagnostic[1024];", types);

        Assert.Contains("jyppx_trt8_execution_context_get_callback_state_snapshot", header8);
        Assert.Contains("jyppx_trt8_execution_context_clear_callback_state", header8);
        Assert.Contains("jyppx_trt10_execution_context_get_callback_state_snapshot", header10);
        Assert.Contains("jyppx_trt10_execution_context_clear_callback_state", header10);
        Assert.Contains("jyppx_trt11_execution_context_get_callback_state_snapshot", header11);
        Assert.Contains("jyppx_trt11_execution_context_clear_callback_state", header11);

        Assert.Contains("execution-context-callback-state-snapshot copied callback boundary state", nativeSource);
        Assert.Contains("context_payload->getOutputAllocator(tensor_name) != nullptr", nativeSource);
        Assert.Contains("context_payload->getTemporaryStorageAllocator() != nullptr", nativeSource);
        Assert.Contains("context_payload->getDebugListener() != nullptr", nativeSource);
        Assert.Contains("context_payload->setOutputAllocator(tensor_name, nullptr)", nativeSource);
        Assert.Contains("context_payload->setTemporaryStorageAllocator(nullptr)", nativeSource);
        Assert.Contains("context_payload->setDebugListener(nullptr)", nativeSource);
        Assert.Contains("copy_callback_interface_info_to_state_with_seh_guard", nativeSource);
        Assert.Contains("TensorRT 8 callback state snapshot does not include debug listener", nativeSource);
        Assert.DoesNotContain("reallocateOutput", nativeSource);
        Assert.DoesNotContain("notifyShape", nativeSource);
        Assert.DoesNotContain("processDebugTensor", nativeSource);
        Assert.DoesNotContain("reinterpret_cast<uintptr_t>", nativeSource);

        Assert.Contains("#include \"../common/execution_context_callback_state_snapshot.inc\"", trt8Api);
        Assert.Contains("#include \"../common/execution_context_callback_state_snapshot.inc\"", trt10Api);
        Assert.Contains("#include \"../common/execution_context_callback_state_snapshot.inc\"", trt11Api);

        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_execution_context_get_callback_state_snapshot", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_execution_context_get_callback_state_snapshot", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_execution_context_get_callback_state_snapshot", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_callback_state", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_callback_state", interop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_callback_state", interop);
        Assert.Contains("NativeTensorRtExecutionContextCallbackStateInfo", nativeStructs);

        Assert.Contains("public TensorRtExecutionContextCallbackStateSnapshot GetCallbackStateSnapshot", wrapper);
        Assert.Contains("public TensorRtExecutionContextCallbackStateSnapshot ClearCallbackState", wrapper);
        Assert.Contains("CreateCallbackStateSnapshot", wrapper);
        Assert.Contains("BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic)", wrapper);
        Assert.Contains("public sealed class TensorRtExecutionContextCallbackStateSnapshot", snapshotWrapper);
        Assert.Contains("public bool HasOutputAllocator", snapshotWrapper);
        Assert.Contains("public bool OutputAllocatorInterfaceInfoAvailable", snapshotWrapper);
        Assert.Contains("public bool DebugListenerClearSupported", snapshotWrapper);
        Assert.Contains("public BridgeStatusCode LastStatus", snapshotWrapper);
        Assert.Contains("public string Diagnostic", snapshotWrapper);
        Assert.DoesNotContain("public IntPtr", snapshotWrapper + wrapper);
        Assert.DoesNotContain("public nint", snapshotWrapper + wrapper);

        Assert.Contains("ExecutionContextCallbackStateSnapshot=GetCallbackStateSnapshot;ClearCallbackState;TensorRtExecutionContextCallbackStateSnapshot", smokeProgram);
        Assert.Contains("context.GetCallbackStateSnapshot(outputTensorName)", smokeProgram);
        Assert.Contains("context.ClearCallbackState(outputTensorName)", smokeProgram);
        Assert.Contains("execution-context-callback-state-snapshot", bridgeConsumer);
        Assert.Contains("hasExecutionContextCallbackStateSnapshot", readiness);

        Assert.Contains("trt8-output-allocator-notify-shape-deferred", trt8Deferred);
        Assert.Contains("trt8-output-allocator-reallocate-output-deferred", trt8Deferred);
        Assert.Contains("trt10-output-allocator-notify-shape-deferred", trt10Deferred);
        Assert.Contains("trt10-output-allocator-reallocate-output-deferred", trt10Deferred);
        Assert.Contains("trt10-debug-listener-process-debug-tensor-deferred", trt10Deferred);
        Assert.Contains("trt11-output-allocator-notify-shape-deferred", trt11Deferred);
        Assert.Contains("trt11-output-allocator-reallocate-output-deferred", trt11Deferred);
        Assert.Contains("trt11-debug-listener-process-debug-tensor-deferred", trt11Deferred);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
