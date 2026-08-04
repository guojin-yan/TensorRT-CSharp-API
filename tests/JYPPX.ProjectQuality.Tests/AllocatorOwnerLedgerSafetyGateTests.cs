using System;
using System.IO;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class AllocatorOwnerLedgerSafetyGateTests
{
    [Fact]
    public void SafetyGateCopiesLedgerLifecycleDiagnosticsWithoutRuntimeProof()
    {
        using TensorRtAllocatorCallbackOwner owner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success($"ledger-safety:{request.Reason}:{request.Size}:{request.Alignment}"));

        TensorRtAllocatorLedgerSafetyGateResult gate = TensorRtAllocatorLedgerSafetyGate.Evaluate(
            owner,
            TensorRtApiLine.TensorRt11,
            new TensorRtAllocatorDryRunRequest(16384, 1024, "quality-ledger-safety"),
            "IGpuAllocator",
            0UL);

        Assert.Equal("allocator-owner-ledger-safety-gate", gate.EvidenceKind);
        Assert.Equal("sync-allocator-ledger-safety", gate.CallbackKind);
        Assert.Equal("ledger-safety-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.False(gate.IsAttached);
        Assert.True(gate.ManagedKeepAliveReady);
        Assert.False(gate.DisposeReleaseReady);
        Assert.True(gate.CallbackStatePinned);
        Assert.True(gate.DelegatePinned);
        Assert.Equal(1, gate.InvocationCount);
        Assert.Equal(0, gate.FailureCount);
        Assert.Equal(0, gate.InFlightCallbackCount);
        Assert.True(gate.MaxInFlightCallbackCount >= 1);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.False(gate.LineSpecificAttachDetachReady);
        Assert.False(gate.DevicePointerLedgerRuntimeReady);
        Assert.False(gate.StreamLifetimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.True(gate.BlockedPrerequisiteCount > 0);
        Assert.Contains("RealCallbackRuntime=False", gate.Diagnostic);
        Assert.Contains("IsRealCallbackRuntimeProof=False", gate.Diagnostic);

        if (gate.NativeLedgerAvailable)
        {
            Assert.True(gate.NativeLedgerDesignReady);
            Assert.True(gate.StateTransitionCount >= 2UL);
            Assert.Equal(gate.LedgerAllocationCount, gate.LedgerReleaseCount);
            Assert.Equal(0UL, gate.LedgerFailureCount);
            Assert.False(gate.HasLiveAllocation);
        }
        else
        {
            Assert.False(gate.NativeLedgerDesignReady);
            Assert.Contains("native ledger diagnostic unavailable", gate.NativeLedgerDiagnostic);
        }

        owner.Dispose();
        TensorRtAllocatorLedgerSafetyGateResult disposed =
            TensorRtAllocatorLedgerSafetyGate.GetSnapshot(owner, "post-dispose");

        Assert.Equal("allocator-owner-ledger-safety-gate", disposed.EvidenceKind);
        Assert.False(disposed.ManagedKeepAliveReady);
        Assert.True(disposed.DisposeReleaseReady);
        Assert.False(disposed.CallbackStatePinned);
        Assert.False(disposed.DelegatePinned);
        Assert.True(disposed.DisposeRequested);
        Assert.True(disposed.ReleaseHookCount >= 1);
        Assert.False(disposed.NativeLedgerAvailable);
        Assert.False(disposed.CanAttemptRuntimeProof);
        Assert.True(disposed.RuntimeProofBlocked);

        TensorRtAllocatorLedgerSafetyGateResult trt10Snapshot =
            TensorRtAllocatorLedgerSafetyGate.GetSnapshot(owner, TensorRtApiLine.TensorRt10, "trt10-post-dispose");
        Assert.Equal(TensorRtApiLine.TensorRt10, trt10Snapshot.Line);
    }

    [Fact]
    public void SafetyGateSourceDocsSmokeAndReadinessKeepProofBoundary()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtAllocatorLedgerSafetyGate.cs");
        string ownerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtAllocatorCallbackOwner.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "allocator-owner-ledger-safety-gate.md");
        string callbackDesign = ReadSource("docs", "articles", "zh-cn", "allocator-callback-owner-design.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtAllocatorLedgerSafetyGate", source);
        Assert.Contains("public readonly struct TensorRtAllocatorLedgerSafetyGateResult", source);
        Assert.Contains("Evaluate", source);
        Assert.Contains("GetSnapshot", source);
        Assert.Contains("allocator-owner-ledger-safety-gate", source);
        Assert.Contains("RuntimeEvidenceKind => \"ledger-safety-gate\"", source);
        Assert.Contains("RealCallbackRuntime => false", source);
        Assert.Contains("IsRealCallbackRuntimeProof => false", source);
        Assert.Contains("ManagedKeepAliveReady", source);
        Assert.Contains("DisposeReleaseReady", source);
        Assert.Contains("NativeLedgerDesignReady", source);
        Assert.Contains("CanAttemptRuntimeProof", source);
        Assert.Contains("RuntimeProofBlocked", source);
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public nint", source);

        Assert.Contains("RunInternalSyncAllocatorRuntimePrototype", ownerSource);
        Assert.Contains("RunNativeStateLedgerDryRunDiagnostic", ownerSource);
        Assert.Contains("AllocatorOwnerLedgerSafetyGate=", smokeProgram);
        Assert.Contains("AllocatorOwnerLedgerSafetyGateDispose=", smokeProgram);
        Assert.Contains("allocator-owner-ledger-safety-gate", smokeProgram);
        Assert.Contains("ManagedKeepAliveReady", smokeProgram);
        Assert.Contains("DisposeReleaseReady", smokeProgram);
        Assert.Contains("NativeLedgerDesignReady", smokeProgram);

        Assert.Contains("allocator-owner-ledger-safety-gate", gateDoc);
        Assert.Contains("TensorRtAllocatorLedgerSafetyGate", gateDoc);
        Assert.Contains("TensorRtAllocatorLedgerSafetyGateResult", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=ledger-safety-gate", gateDoc);
        Assert.Contains("RealCallbackRuntime=False", gateDoc);
        Assert.Contains("IsRealCallbackRuntimeProof=False", gateDoc);
        Assert.Contains("not proof", gateDoc);

        Assert.Contains("allocator-owner-ledger-safety-gate.md", docsIndex);
        Assert.Contains("allocator-owner-ledger-safety-gate.md", docsToc);
        Assert.Contains("allocator-owner-ledger-safety-gate", callbackDesign);
        Assert.Contains("allocator-owner-ledger-safety-gate", trampolineGate);
        Assert.Contains("allocator-owner-ledger-safety-gate", schema);
        Assert.Contains("allocator-owner-ledger-safety-gate", latest);
        Assert.Contains("allocator-owner-ledger-safety-gate", smokeReadme);

        Assert.Contains("New-AllocatorOwnerLedgerSafetyGateEvidence", readiness);
        Assert.Contains("allocatorOwnerLedgerSafetyGate", readiness);
        Assert.Contains("safety-gate-ready", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("allocator-owner-ledger-safety-gate", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorLedgerSafetyGate", bridgeConsumer);
        Assert.Contains("TensorRtAllocatorLedgerSafetyGateResult", bridgeConsumer);
        Assert.Contains("allocator-owner-ledger-safety-gate", packageConsumer);

        Assert.Contains("\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IGpuAsyncAllocator\",\"allocateAsync\",\"IGpuAsyncAllocator::allocateAsync\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string root = FindRepositoryRoot();
        string path = Path.Combine(new[] { root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }

    private static string FindRepositoryRoot()
    {
        string directory = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(directory))
        {
            if (File.Exists(Path.Combine(directory, "TensorRtSharp.sln")))
            {
                return directory;
            }

            DirectoryInfo? parent = Directory.GetParent(directory);
            if (parent == null)
            {
                break;
            }

            directory = parent.FullName;
        }

        throw new InvalidOperationException("Repository root was not found.");
    }
}
