using System;
using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CallbackAllocatorReadinessSnapshotTests
{
    [Fact]
    public void ReadinessAggregatesExistingGatesWithoutPromotingRuntimeProof()
    {
        using TensorRtAllocatorCallbackOwner allocatorOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success("quality-readiness:" + request.Reason));
        TensorRtAllocatorLedgerSafetyGateResult allocatorGate = TensorRtAllocatorLedgerSafetyGate.Evaluate(
            allocatorOwner,
            TensorRtApiLine.TensorRt11,
            new TensorRtAllocatorDryRunRequest(32768UL, 256UL, "quality-readiness-allocator"),
            "IGpuAllocator",
            0UL);

        using TensorRtOutputAllocatorCallbackOwner outputOwner = new TensorRtOutputAllocatorCallbackOwner();
        _ = outputOwner.RunDesignDiagnostic(
            TensorRtApiLine.TensorRt11,
            new TensorRtOutputAllocatorCallbackRequest(
                "quality_readiness_output",
                4096UL,
                256UL,
                new long[] { 1, 1000 },
                "quality-readiness-output",
                hasCurrentMemory: true),
            0UL);
        outputOwner.Dispose();
        TensorRtOutputAllocatorRuntimeProofPrecheckResult outputPrecheck =
            TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(outputOwner.GetSnapshot("post-dispose-readiness"));

        using TensorRtDebugListenerCallbackOwner debugOwner = new TensorRtDebugListenerCallbackOwner();
        _ = debugOwner.RunDesignDiagnostic(
            TensorRtApiLine.TensorRt11,
            new TensorRtDebugListenerCallbackRequest(
                "quality_readiness_debug",
                TensorRtDataType.Float,
                TensorRtTensorLocation.Device,
                new long[] { 1, 3, 224, 224 },
                "quality-readiness-debug",
                isInput: true,
                isExecutionTensor: true));
        debugOwner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult debugPrecheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(debugOwner.GetSnapshot("post-dispose-readiness"));

        TensorRtCallbackAllocatorReadinessSnapshot readiness = TensorRtCallbackAllocatorReadiness.Evaluate(
            allocatorGate,
            outputPrecheck,
            debugPrecheck);

        Assert.Equal("callback-allocator-readiness-snapshot", readiness.EvidenceKind);
        Assert.Equal("managed-readiness", readiness.RuntimeEvidenceKind);
        Assert.False(readiness.RealCallbackRuntime);
        Assert.False(readiness.IsRealCallbackRuntimeProof);
        Assert.True(readiness.LoggerCallbackReady);
        Assert.True(readiness.ProfilerCallbackReady);
        Assert.True(readiness.ProgressMonitorCallbackReady);
        Assert.True(readiness.AllocatorOwnerDryRunReady);
        Assert.True(readiness.OutputAllocatorOwnerDesignReady);
        Assert.True(readiness.OutputAllocatorRuntimeGateReady);
        Assert.True(readiness.DebugListenerOwnerDesignReady);
        Assert.True(readiness.DebugListenerRuntimeProofPrecheckReady);
        Assert.False(readiness.RealCallbackInvocationProofReady);
        Assert.False(readiness.IsRuntimeInvocationProofComplete);
        Assert.True(readiness.RuntimeProofBlocked);
        Assert.True(readiness.BlockedReasonCount >= outputPrecheck.BlockedPrerequisiteCount);
        Assert.Contains("RealCallbackRuntime=False", readiness.Summary);
        Assert.Contains("IsRuntimeInvocationProofComplete=False", readiness.Summary);
        Assert.Contains("RuntimeProofBlocked=True", readiness.Summary);
    }

    [Fact]
    public void ReadinessPublicSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtCallbackAllocatorReadiness),
            typeof(TensorRtCallbackAllocatorReadinessSnapshot),
            typeof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary)
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
    public void SmokeConsumerAndDocsExposeReadinessSnapshotBoundary()
    {
        string readinessSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtCallbackAllocatorReadiness.cs");
        string safeControlSummarySource = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContextCallbackAllocatorSafeControlSummary.cs");
        string executionContextDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string boundaryGuide = ReadSource("docs", "articles", "zh-cn", "callback-allocator-boundary-guide.md");
        string roadmap = ReadSource("docs", "articles", "zh-cn", "callback-allocator-safety-bridge-roadmap.md");

        Assert.Contains("public static class TensorRtCallbackAllocatorReadiness", readinessSource);
        Assert.Contains("public sealed class TensorRtCallbackAllocatorReadinessSnapshot", readinessSource);
        Assert.Contains("public static TensorRtCallbackAllocatorReadinessSnapshot Evaluate", readinessSource);
        Assert.Contains("public bool IsPublishSafeForManagedCallbacks", readinessSource);
        Assert.Contains("public bool IsRuntimeInvocationProofComplete", readinessSource);
        Assert.Contains("public int BlockedReasonCount", readinessSource);
        Assert.Contains("not proof that TensorRT has invoked allocator, output allocator, or debug listener callbacks", readinessSource);
        Assert.DoesNotContain("public IntPtr", readinessSource);
        Assert.DoesNotContain("public nint", readinessSource);

        Assert.Contains("public sealed class TensorRtExecutionContextCallbackAllocatorSafeControlSummary", safeControlSummarySource);
        Assert.Contains("public TensorRtExecutionContextCallbackAllocatorSafeControlSummary GetCallbackAllocatorSafeControlSummary", executionContextDiagnostics);
        Assert.Contains("copied metadata only", safeControlSummarySource + executionContextDiagnostics);
        Assert.Contains("Borrowed pointer not exposed/owned", safeControlSummarySource + executionContextDiagnostics);
        Assert.Contains("no callback invocation", safeControlSummarySource + executionContextDiagnostics);
        Assert.Contains("not runtime proof", safeControlSummarySource + executionContextDiagnostics);
        Assert.Contains("public int CopiedInterfaceInfoCount", safeControlSummarySource);
        Assert.Contains("public int DiagnosticCount", safeControlSummarySource);
        Assert.Contains("public bool PointerFreeSurfaceReady", safeControlSummarySource);
        Assert.Contains("public bool IsRuntimeInvocationProofComplete", safeControlSummarySource);
        Assert.DoesNotContain("public IntPtr", safeControlSummarySource + executionContextDiagnostics);
        Assert.DoesNotContain("public nint", safeControlSummarySource + executionContextDiagnostics);

        Assert.Contains("CallbackAllocatorSafeControlSummary=", smokeProgram);
        Assert.Contains("FormatCallbackAllocatorSafeControlSummary", smokeProgram);
        Assert.Contains("context.GetCallbackAllocatorSafeControlSummary(outputTensorName)", smokeProgram);
        Assert.Contains("execution-context-callback-allocator-safe-control-summary", smokeProgram);
        Assert.Contains("copied-metadata-only;pointer-free;not-runtime-proof", smokeProgram);

        Assert.Contains("CallbackAllocatorReadinessSnapshot=", smokeProgram);
        Assert.Contains("FormatCallbackAllocatorReadinessSnapshot", smokeProgram);
        Assert.Contains("IsPublishSafeForManagedCallbacks", smokeProgram);
        Assert.Contains("IsRuntimeInvocationProofComplete", smokeProgram);
        Assert.Contains("RealCallbackInvocationProofReady", smokeProgram);

        Assert.Contains("TensorRtCallbackAllocatorReadiness.Evaluate", bridgeConsumer);
        Assert.Contains("TensorRtCallbackAllocatorReadinessSnapshot", bridgeConsumer);
        Assert.Contains("callbackAllocatorReadinessSnapshot", bridgeConsumer);
        Assert.Contains("callbackAllocatorReadinessSummary", bridgeConsumer);
        Assert.Contains("TensorRtExecutionContextCallbackAllocatorSafeControlSummary", bridgeConsumer);
        Assert.Contains("contextCallbackAllocatorSafeControlSummary", bridgeConsumer);
        Assert.Contains("execution-context-callback-allocator-safe-control-summary", bridgeConsumer);
        Assert.Contains("GetCallbackAllocatorSafeControlSummary", bridgeConsumer);

        Assert.Contains("TensorRtCallbackAllocatorReadinessSnapshot", boundaryGuide);
        Assert.Contains("managed readiness", boundaryGuide);
        Assert.Contains("不能作为真实 TensorRT callback runtime proof", boundaryGuide);
        Assert.Contains("TensorRtExecutionContextCallbackAllocatorSafeControlSummary", boundaryGuide);
        Assert.Contains("copied metadata only", boundaryGuide);
        Assert.Contains("TensorRtCallbackAllocatorReadinessSnapshot", roadmap);
        Assert.Contains("BlockedReasonCount", roadmap);
        Assert.Contains("IsRuntimeInvocationProofComplete", roadmap);
        Assert.Contains("TensorRtExecutionContextCallbackAllocatorSafeControlSummary", roadmap);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
