using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputAllocatorRuntimeProofPrecheckTests
{
    [Fact]
    public void PrecheckCopiesOwnerDesignStateAndStaysBlockedBeforeRuntimeProof()
    {
        using TensorRtOutputAllocatorCallbackOwner owner = new TensorRtOutputAllocatorCallbackOwner();
        TensorRtOutputAllocatorCallbackRequest request = new TensorRtOutputAllocatorCallbackRequest(
            "quality_output_allocator",
            8192UL,
            256UL,
            new long[] { 1, 16, 32 },
            "quality-output-allocator-runtime-proof-precheck",
            hasCurrentMemory: true);

        TensorRtOutputAllocatorCallbackOwnerSnapshot diagnostic =
            owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request, 0UL);
        Assert.Equal(1, diagnostic.NotifyShapeCount);
        Assert.Equal(1, diagnostic.ReallocateOutputCount);

        owner.Dispose();
        TensorRtOutputAllocatorCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtOutputAllocatorRuntimeProofPrecheckResult precheck =
            TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(disposed);

        Assert.Equal("output-allocator-runtime-proof-precheck", precheck.EvidenceKind);
        Assert.Equal("output-allocator", precheck.CallbackKind);
        Assert.Equal("runtime-gate", precheck.RuntimeEvidenceKind);
        Assert.False(precheck.RealCallbackRuntime);
        Assert.False(precheck.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, precheck.Line);
        Assert.True(precheck.LineSupportsOutputAllocator);
        Assert.True(precheck.OwnerDesignReady);
        Assert.Equal(disposed.NativeLedgerAvailable, precheck.NativeLedgerDesignReady);
        Assert.True(precheck.DisposeReleaseReady);
        Assert.True(precheck.PointerFreeSurfaceReady);
        Assert.True(precheck.AttachDetachDesignGateReady);
        Assert.False(precheck.AttachControlAvailable);
        Assert.True(precheck.DetachClearControlAvailable);
        Assert.True(precheck.ManagedOwnerStateMachineReady);
        Assert.False(precheck.LineSpecificAttachDetachReady);
        Assert.False(precheck.StableNativeOwnerAddressReady);
        Assert.False(precheck.NoThrowNativeVTableReady);
        Assert.False(precheck.NativeVTableReady);
        Assert.False(precheck.DevicePointerLedgerRuntimeReady);
        Assert.False(precheck.StreamLifetimeReady);
        Assert.True(precheck.OutputBufferOwnershipSafetyGateReady);
        Assert.False(precheck.OutputBufferOwnershipRuntimeReady);
        Assert.False(precheck.CurrentMemoryReusePolicyReady);
        Assert.True(precheck.BorrowedPointerEscapeBlocked);
        Assert.False(precheck.OwnedDevicePointerReleasePolicyReady);
        Assert.False(precheck.ShapeNotificationOrderingReady);
        Assert.False(precheck.ReallocateOutputRuntimeReady);
        Assert.False(precheck.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(precheck.DeferredRowsStillRequired);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Equal("precheck-blocked", precheck.Status);
        Assert.True(precheck.BlockedPrerequisiteCount >= 4);
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("setOutputAllocator", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("stable address", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("device pointer", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("stream lifetime", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("output buffer ownership", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("currentMemory reuse policy", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("owned device pointer release policy", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("reallocateOutput runtime callback", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
        Assert.Contains("RealCallbackRuntime=False", precheck.Diagnostic);
        Assert.Contains("IsRealCallbackRuntimeProof=False", precheck.Diagnostic);
        Assert.Contains("AttachDetachDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("AttachControlAvailable=False", precheck.Diagnostic);
        Assert.Contains("DetachClearControlAvailable=True", precheck.Diagnostic);
        Assert.Contains("OutputBufferOwnershipRuntimeReady=False", precheck.Diagnostic);
        Assert.Contains("OutputBufferOwnershipSafetyGateReady=True", precheck.Diagnostic);
        Assert.Contains("CurrentMemoryReusePolicyReady=False", precheck.Diagnostic);
        Assert.Contains("BorrowedPointerEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("ReallocateOutputRuntimeReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void NativeLedgerUnavailableSnapshotDoesNotPromoteRuntimeProof()
    {
        using TensorRtOutputAllocatorCallbackOwner owner = new TensorRtOutputAllocatorCallbackOwner();
        TensorRtOutputAllocatorCallbackOwnerSnapshot snapshot = owner.GetSnapshot("pre-native-ledger");

        TensorRtOutputAllocatorRuntimeProofPrecheckResult precheck =
            TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(snapshot);

        Assert.False(precheck.OwnerDesignReady);
        Assert.False(precheck.NativeLedgerDesignReady);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.False(precheck.IsRealCallbackRuntimeProof);
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("native allocator owner ledger", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicPrecheckSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtOutputAllocatorRuntimeProofPrecheck),
            typeof(TensorRtOutputAllocatorRuntimeProofPrecheckResult)
        };

        foreach (Type type in publicTypes)
        {
            foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
            {
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }

            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
            {
                Assert.NotEqual(typeof(IntPtr), property.PropertyType);
                Assert.NotEqual(typeof(UIntPtr), property.PropertyType);
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                Assert.NotEqual(typeof(IntPtr), method.ReturnType);
                Assert.NotEqual(typeof(UIntPtr), method.ReturnType);
                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }
        }
    }

    [Fact]
    public void ReadinessSmokeAndDocsKeepPrecheckSeparateFromRuntimeProof()
    {
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputAllocatorRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-runtime-proof-precheck.md");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-callback-owner-design.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtOutputAllocatorRuntimeProofPrecheck", precheckSource);
        Assert.Contains("public readonly struct TensorRtOutputAllocatorRuntimeProofPrecheckResult", precheckSource);
        Assert.Contains("RuntimeEvidenceKind => \"runtime-gate\"", precheckSource);
        Assert.Contains("RealCallbackRuntime => false", precheckSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", precheckSource);
        Assert.Contains("TensorRtOutputAllocatorAttachDetachDesignGate", precheckSource);
        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGate", precheckSource);
        Assert.Contains("AttachDetachDesignGateReady", precheckSource);
        Assert.Contains("LineSupportsOutputAllocator", precheckSource);
        Assert.Contains("AttachControlAvailable", precheckSource);
        Assert.Contains("DetachClearControlAvailable", precheckSource);
        Assert.Contains("ManagedOwnerStateMachineReady", precheckSource);
        Assert.Contains("LineSpecificAttachDetachReady", precheckSource);
        Assert.Contains("StableNativeOwnerAddressReady", precheckSource);
        Assert.Contains("NoThrowNativeVTableReady", precheckSource);
        Assert.Contains("NativeVTableReady", precheckSource);
        Assert.Contains("DevicePointerLedgerRuntimeReady => false", precheckSource);
        Assert.Contains("StreamLifetimeReady => false", precheckSource);
        Assert.Contains("OutputBufferOwnershipSafetyGateReady", precheckSource);
        Assert.Contains("OutputBufferOwnershipRuntimeReady", precheckSource);
        Assert.Contains("CurrentMemoryReusePolicyReady", precheckSource);
        Assert.Contains("BorrowedPointerEscapeBlocked", precheckSource);
        Assert.Contains("OwnedDevicePointerReleasePolicyReady", precheckSource);
        Assert.Contains("ShapeNotificationOrderingReady", precheckSource);
        Assert.Contains("ReallocateOutputRuntimeReady", precheckSource);
        Assert.Contains("CanAttemptRuntimeProof", precheckSource);
        Assert.DoesNotContain("public IntPtr", precheckSource);
        Assert.DoesNotContain("public nint", precheckSource);

        Assert.Contains("output-allocator-runtime-proof-precheck", smokeProgram);
        Assert.Contains("output-allocator-attach-detach-design-gate", smokeProgram);
        Assert.Contains("output-buffer-ownership-safety-gate", smokeProgram);
        Assert.Contains("OutputAllocatorRuntimeProofPrecheck=", smokeProgram);
        Assert.Contains("OutputAllocatorAttachDetachDesignGate=", smokeProgram);
        Assert.Contains("OutputBufferOwnershipSafetyGate=", smokeProgram);
        Assert.Contains("AttachControlAvailable", smokeProgram);
        Assert.Contains("DetachClearControlAvailable", smokeProgram);
        Assert.Contains("StableNativeOwnerAddressReady", smokeProgram);
        Assert.Contains("NoThrowNativeVTableReady", smokeProgram);
        Assert.Contains("OutputBufferOwnershipRuntimeReady", smokeProgram);
        Assert.Contains("CurrentMemoryReusePolicyReady", smokeProgram);
        Assert.Contains("ReallocateOutputRuntimeReady", smokeProgram);
        Assert.Contains("CanAttemptRuntimeProof", smokeProgram);
        Assert.Contains("RuntimeProofBlocked", smokeProgram);

        Assert.Contains("outputAllocatorRuntimeProofPrecheck", readiness);
        Assert.Contains("New-OutputAllocatorRuntimeProofPrecheckEvidence", readiness);
        Assert.Contains("outputAllocatorAttachDetachDesignGate", readiness);
        Assert.Contains("New-OutputAllocatorAttachDetachDesignGateEvidence", readiness);
        Assert.Contains("outputBufferOwnershipSafetyGate", readiness);
        Assert.Contains("New-OutputBufferOwnershipSafetyGateEvidence", readiness);
        Assert.Contains("runtime-gate-precheck", readiness);
        Assert.Contains("precheck-ready", readiness);
        Assert.Contains("AttachControlAvailable=False", readiness);
        Assert.Contains("DetachClearControlAvailable", readiness);
        Assert.Contains("StableNativeOwnerAddressReady", readiness);
        Assert.Contains("NoThrowNativeVTableReady", readiness);
        Assert.Contains("OutputBufferOwnershipRuntimeReady", readiness);
        Assert.Contains("OutputBufferOwnershipSafetyGateReady", readiness);
        Assert.Contains("CurrentMemoryReusePolicyReady", readiness);
        Assert.Contains("ReallocateOutputRuntimeReady", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("output-allocator-runtime-proof-precheck", packageConsumer);
        Assert.Contains("output-allocator-attach-detach-design-gate", packageConsumer);
        Assert.Contains("output-buffer-ownership-safety-gate", packageConsumer);
        Assert.Contains("output-allocator-runtime-proof-precheck", bridgeConsumer);
        Assert.Contains("output-allocator-attach-detach-design-gate", bridgeConsumer);
        Assert.Contains("output-buffer-ownership-safety-gate", bridgeConsumer);
        Assert.Contains("TensorRtOutputAllocatorRuntimeProofPrecheck", bridgeConsumer);
        Assert.Contains("TensorRtOutputAllocatorAttachDetachDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGate", bridgeConsumer);

        Assert.Contains("output-allocator-runtime-proof-precheck", precheckDoc);
        Assert.Contains("RuntimeEvidenceKind=runtime-gate", precheckDoc);
        Assert.Contains("AttachDetachDesignGateReady=True", precheckDoc);
        Assert.Contains("CanAttemptRuntimeProof=False", precheckDoc);
        Assert.Contains("RuntimeProofBlocked=True", precheckDoc);
        Assert.Contains("AttachControlAvailable=False", precheckDoc);
        Assert.Contains("DetachClearControlAvailable=True", precheckDoc);
        Assert.Contains("StableNativeOwnerAddressReady=False", precheckDoc);
        Assert.Contains("NoThrowNativeVTableReady=False", precheckDoc);
        Assert.Contains("OutputBufferOwnershipRuntimeReady=False", precheckDoc);
        Assert.Contains("OutputBufferOwnershipSafetyGateReady=True", precheckDoc);
        Assert.Contains("CurrentMemoryReusePolicyReady=False", precheckDoc);
        Assert.Contains("ReallocateOutputRuntimeReady=False", precheckDoc);
        Assert.Contains("setOutputAllocator", precheckDoc);
        Assert.Contains("not proof", precheckDoc);
        Assert.Contains("output-allocator-attach-detach-design-gate", precheckDoc);
        Assert.Contains("output-buffer-ownership-safety-gate", precheckDoc);
        Assert.Contains("output-allocator-runtime-proof-precheck", designDoc);
        Assert.Contains("output-allocator-attach-detach-design-gate", designDoc);
        Assert.Contains("output-buffer-ownership-safety-gate", designDoc);
        Assert.Contains("output-allocator-runtime-proof-precheck", trampolineGate);
        Assert.Contains("output-allocator-attach-detach-design-gate", trampolineGate);
        Assert.Contains("output-buffer-ownership-safety-gate", trampolineGate);
        Assert.Contains("output-allocator-runtime-proof-precheck", schema);
        Assert.Contains("output-allocator-attach-detach-design-gate", schema);
        Assert.Contains("output-buffer-ownership-safety-gate", schema);
        Assert.Contains("output-allocator-runtime-proof-precheck", latest);
        Assert.Contains("output-allocator-attach-detach-design-gate", latest);
        Assert.Contains("output-buffer-ownership-safety-gate", latest);
        Assert.Contains("outputAllocatorRuntimeProofPrecheck", runtimeSplitReadme);
        Assert.Contains("outputAllocatorAttachDetachDesignGate", runtimeSplitReadme);
        Assert.Contains("outputBufferOwnershipSafetyGate", runtimeSplitReadme);
        Assert.Contains("output-allocator-runtime-proof-precheck", smokeReadme);
        Assert.Contains("output-allocator-attach-detach-design-gate", smokeReadme);
        Assert.Contains("output-buffer-ownership-safety-gate", smokeReadme);

        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
