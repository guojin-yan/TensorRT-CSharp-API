using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerCallbackProofGapReportTests
{
    [Fact]
    public void GapReportDefaultsToPointerFreeBlockedAndNonProof()
    {
        TensorRtDebugListenerCallbackProofGapReportResult report = CreateReport();

        Assert.Equal("debug-listener-callback-proof-gap-report", report.EvidenceKind);
        Assert.Equal("proof-gap-report", report.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", report.CallbackKind);
        Assert.Equal(TensorRtApiLine.TensorRt11, report.Line);
        Assert.Equal(11, report.TensorRtLine);
        Assert.False(report.RealCallbackRuntime);
        Assert.False(report.IsRealCallbackRuntimeProof);
        Assert.True(report.NonNullAttachStillDisabled);
        Assert.False(report.NativeAttachEntryReady);
        Assert.True(report.NativeVTableInstallBlocked);
        Assert.True(report.NoThrowCallbackEntryReady);
        Assert.True(report.ExceptionStatusMappingReady);
        Assert.True(report.InFlightAccountingReady);
        Assert.True(report.BorrowedDebugTensorMetadataCopied);
        Assert.True(report.DetachRollbackReady);
        Assert.False(report.ProcessDebugTensorRuntimeInvoked);
        Assert.False(report.FullPackageConsumerRuntimeProofReady);
        Assert.True(report.PointerFreeSurfaceReady);
        Assert.False(report.AttemptedNoInvocation);
        Assert.Equal(0, report.InvocationCount);
        Assert.Equal(0, report.FailureCount);
        Assert.Equal(0, report.InFlightCallbackCount);
        Assert.False(report.CanAttemptRuntimeProof);
        Assert.False(report.CanPromoteRealCallbackRuntime);
        Assert.True(report.RuntimeProofBlocked);
        Assert.True(report.DeferredRowsStillRequired);
        Assert.Equal("blocked", report.Status);
        Assert.True(report.GapReasonCount > 0);
        Assert.Contains("non-null", report.PrimaryGapReason, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("non-null-attach-disabled", report.RuntimeProofBlockerCategory);
        Assert.True(report.PackageConsumerRuntimeProofRequired);
        Assert.True(report.RuntimeInvocationRequired);
        Assert.Equal("copied-preflight-smoke-trampoline-proof-gate", report.EvidenceSource);
        Assert.Equal("enable-and-verify-non-null-debug-listener-attach-under-version-guards", report.NextOwnerAction);
        Assert.Contains(report.GapReasons, reason => reason.Contains("TensorRT has not invoked", StringComparison.OrdinalIgnoreCase));
        Assert.Contains("RuntimeEvidenceKind=proof-gap-report", report.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("IsRealCallbackRuntimeProof=False", report.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("RuntimeProofBlockerCategory=non-null-attach-disabled", report.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NextOwnerAction=enable-and-verify-non-null-debug-listener-attach-under-version-guards", report.Diagnostic, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicGapReportSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerCallbackProofGapReport));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerCallbackProofGapReportResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerCallbackProofGapReport.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("PointerFreeSurfaceReady", source);
        Assert.Contains("proof-gap-report", source);
        Assert.Contains("debug-listener-callback-proof-gap-report", source);
        Assert.Contains("PrimaryGapReason", source);
        Assert.Contains("RuntimeProofBlockerCategory", source);
        Assert.Contains("NextOwnerAction", source);
    }

    [Fact]
    public void SourceSmokePackageReadinessAndDocsCarryGapReportNonProofMarkers()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerCallbackProofGapReport.cs");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-callback-proof-gap-report.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");

        AssertGapReportMarkers(source);
        AssertGapReportMarkers(smoke);
        AssertGapReportMarkers(packageConsumer);
        AssertGapReportMarkers(bridgeConsumer);
        AssertGapReportMarkers(readiness);
        AssertGapReportMarkers(doc);
        AssertGapReportMarkers(schema);
        AssertGapReportMarkers(smokeReadme);
        AssertGapReportMarkers(runtimeSplitReadme);
        AssertActionableGapMarkers(source);
        AssertActionableGapMarkers(smoke);
        AssertActionableGapMarkers(bridgeConsumer);
        AssertActionableGapMarkers(readiness);
        AssertActionableGapMarkers(doc);
        AssertActionableGapMarkers(schema);
        Assert.Contains("IsRealCallbackRuntimeProof=False", doc);
        Assert.Contains("IsRealCallbackRuntimeProof=False", schema);
        Assert.Contains("IsRealCallbackRuntimeProof=False", smokeReadme);
        Assert.Contains("IsRealCallbackRuntimeProof=False", runtimeSplitReadme);
        Assert.Contains("DebugListenerCallbackProofGapReport=", smoke);
        Assert.Contains("DebugListenerCallbackProofGapReport=", packageConsumer);
        Assert.Contains("New-DebugListenerCallbackProofGapReportEvidence", readiness);
        Assert.Contains("debugListenerCallbackProofGapReport", readiness);
        Assert.Contains("debug-listener-callback-proof-gap-report.md", toc);
        Assert.Contains("debug-listener-callback-proof-gap-report.md", index);
        Assert.Contains("NoNonProofCallbackRuntimeMarker", packageConsumer);
        Assert.Contains("NoNonProofCallbackRuntimeMarker", readiness);
        Assert.Contains("callback-owner-closure-matrix", packageConsumer);
        Assert.Contains("callback-owner-closure-matrix", readiness);
    }

    private static TensorRtDebugListenerCallbackProofGapReportResult CreateReport()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_listener_gap_report_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 8, 8 },
            "quality-debug-listener-callback-proof-gap-report",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose-gap-report");
        TensorRtDebugListenerRuntimeProofAttemptPreflightResult attempt =
            TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(snapshot);
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult smoke =
            TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(
                attempt,
                "quality-runtime-key",
                optInEnabled: false,
                fullPackageConsumerReport: false);
        TensorRtDebugListenerNoThrowVTableCallbackStubResult stub =
            TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(snapshot);
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult metadata =
            TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(snapshot);
        TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult trampoline =
            TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(stub, metadata, smoke);
        TensorRtDebugListenerRealCallbackRuntimeProofResult proof =
            TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate(smoke, trampoline);

        return TensorRtDebugListenerCallbackProofGapReport.Evaluate(attempt, smoke, trampoline, proof);
    }

    private static void AssertGapReportMarkers(string text)
    {
        Assert.Contains("debug-listener-callback-proof-gap-report", text);
        Assert.Contains("TensorRtDebugListenerCallbackProofGapReport", text);
        Assert.Contains("TensorRtDebugListenerCallbackProofGapReportResult", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("proof-gap-report", text);
        Assert.Contains("NonNullAttachStillDisabled", text);
        Assert.Contains("NativeAttachEntryReady", text);
        Assert.Contains("NativeVTableInstallBlocked", text);
        Assert.Contains("NoThrowCallbackEntryReady", text);
        Assert.Contains("ExceptionStatusMappingReady", text);
        Assert.Contains("InFlightAccountingReady", text);
        Assert.Contains("BorrowedDebugTensorMetadataCopied", text);
        Assert.Contains("DetachRollbackReady", text);
        Assert.Contains("ProcessDebugTensorRuntimeInvoked", text);
        Assert.Contains("FullPackageConsumerRuntimeProofReady", text);
        Assert.Contains("GapReasonCount", text);
    }

    private static void AssertActionableGapMarkers(string text)
    {
        Assert.Contains("PrimaryGapReason", text);
        Assert.Contains("RuntimeProofBlockerCategory", text);
        Assert.Contains("PackageConsumerRuntimeProofRequired", text);
        Assert.Contains("RuntimeInvocationRequired", text);
        Assert.Contains("EvidenceSource", text);
        Assert.Contains("NextOwnerAction", text);
    }

    private static void AssertNoRawPointerTypes(Type type)
    {
        foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
        {
            foreach (ParameterInfo parameter in constructor.GetParameters())
            {
                AssertNoRawPointerType(parameter.ParameterType);
            }
        }

        foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
        {
            AssertNoRawPointerType(property.PropertyType);
        }

        foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
        {
            AssertNoRawPointerType(method.ReturnType);
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                AssertNoRawPointerType(parameter.ParameterType);
            }
        }
    }

    private static void AssertNoRawPointerType(Type type)
    {
        Assert.NotEqual(typeof(IntPtr), type);
        Assert.NotEqual(typeof(UIntPtr), type);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
