using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerRealCallbackRuntimeProofPreflightBatchTests
{
    [Fact]
    public void RuntimeProofAttemptPreflightStaysPointerFreeBlockedAndNonProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 16, 16 },
            "quality-debug-listener-runtime-proof-attempt-preflight",
            isInput: true,
            isExecutionTensor: true);

        _ = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();

        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));
        TensorRtDebugListenerRuntimeProofAttemptPreflightResult preflight =
            TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(precheck);

        Assert.Equal("debug-listener-runtime-proof-attempt-preflight", preflight.EvidenceKind);
        Assert.Equal("runtime-proof-attempt-preflight", preflight.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", preflight.CallbackKind);
        Assert.False(preflight.RealCallbackRuntime);
        Assert.False(preflight.IsRealCallbackRuntimeProof);
        Assert.False(preflight.CanEnableSetDebugListenerNonNull);
        Assert.False(preflight.CanInstallNativeVTable);
        Assert.False(preflight.CanCallProcessDebugTensorRuntime);
        Assert.False(preflight.CanPromoteRealCallbackRuntime);
        Assert.False(preflight.CanAttemptRuntimeProof);
        Assert.True(preflight.RuntimeProofBlocked);
        Assert.True(preflight.DeferredRowsStillRequired);
        Assert.Equal("runtime-proof-attempt-blocked", preflight.Status);
        Assert.False(preflight.NativeAttachEntryLocated);
        Assert.True(preflight.NonNullAttachStillDisabled);
        Assert.False(preflight.NativeVTableReady);
        Assert.False(preflight.NativeVTableTrampolineReady);
        Assert.False(preflight.ProcessDebugTensorRuntimeReady);
        Assert.False(preflight.FullPackageConsumerRuntimeEvidenceReady);
        Assert.Contains("setDebugListener(non-null)", preflight.ReasonNonNullAttachStillBlocked, StringComparison.Ordinal);
        Assert.Contains("vtable", preflight.ReasonNativeVTableStillBlocked, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("processDebugTensor", preflight.ReasonRuntimeProofStillBlocked, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRealCallbackRuntime=False", preflight.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", preflight.Diagnostic);
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicPreflightSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerRuntimeProofAttemptPreflight));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerRuntimeProofAttemptPreflight.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
    }

    [Fact]
    public void ConsumerScriptsHardenFreshnessAndRealCallbackRuntimePromotion()
    {
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        AssertFreshnessGuard(packageConsumer);
        AssertFreshnessGuard(bridgeConsumer);
        Assert.Contains("Sort-Object LastWriteTime, Version -Descending", packageConsumer);
        Assert.Contains("Managed package appears stale", packageConsumer);
        Assert.Contains("Managed package appears stale", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerRuntimeProofAttemptPreflight", packageConsumer);
        Assert.Contains("TensorRtDebugListenerRuntimeProofAttemptPreflight", bridgeConsumer);

        Assert.Contains("RealCallbackRuntimeNonProofSmokeMarkers", packageConsumer);
        Assert.Contains("Test-SmokeOutputContainsAnyMarker", packageConsumer);
        Assert.Contains("\"debug-listener-runtime-proof-attempt-preflight\"", packageConsumer);
        Assert.Contains("\"attach-bridge-shape-gate\"", packageConsumer);
        Assert.Contains("\"exception-status-gate\"", packageConsumer);
        Assert.Contains("\"inflight-accounting-gate\"", packageConsumer);
        Assert.Contains("\"vtable-scaffold-gate\"", packageConsumer);
        Assert.Contains("\"runtime-gate\"", packageConsumer);
        Assert.Contains("\"RuntimeEvidenceKind=real-callback-runtime\"", packageConsumer);
        Assert.Contains("\"IsRealCallbackRuntimeProof=True\"", packageConsumer);
        Assert.Contains("$hasRuntimeMarker = $combinedSmokeOutput.IndexOf(\"EvidenceKind=real-callback-runtime\"", packageConsumer);
    }

    [Fact]
    public void ReadinessDocsSmokePackageAndDeferredRowsContainPreflightEvidence()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerRuntimeProofAttemptPreflight.cs");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-real-callback-runtime-proof-preflight.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertPreflightMarkers(source);
        AssertPreflightMarkers(smoke);
        AssertPreflightMarkers(readiness);
        AssertPreflightMarkers(bridgeConsumer);
        AssertPreflightMarkers(packageConsumer);
        AssertPreflightMarkers(doc);
        AssertPreflightMarkers(schema);
        AssertPreflightMarkers(latest);
        AssertPreflightMarkers(smokeReadme);
        AssertPreflightMarkers(runtimeSplitReadme);
        AssertPreflightMarkers(toc);
        AssertPreflightMarkers(index);
        Assert.Contains("New-DebugListenerRuntimeProofAttemptPreflightEvidence", readiness);
        Assert.Contains("debugListenerRuntimeProofAttemptPreflight", readiness);
        Assert.Contains("hasDebugListenerRuntimeProofAttemptPreflight", readiness);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertFreshnessGuard(string script)
    {
        Assert.Contains("ManagedPackageFreshnessRequiredMarkers", script);
        Assert.Contains("ManagedPackageFreshnessPackCommand", script);
        Assert.Contains("Get-ManagedPackageXmlSurface", script);
        Assert.Contains("Assert-ManagedPackageFreshness", script);
        Assert.Contains("JYPPX.TensorRtSharp.xml", script);
        Assert.Contains("lib/", script);
        Assert.Contains("/p:UseSharedCompilation=false", script);
        Assert.Contains("Assert-ManagedPackageFreshness -ManagedPackage $managedPackage", script);
    }

    private static void AssertPreflightMarkers(string text)
    {
        Assert.Contains("debug-listener-runtime-proof-attempt-preflight", text);
        Assert.Contains("TensorRtDebugListenerRuntimeProofAttemptPreflight", text);
        Assert.Contains("CanEnableSetDebugListenerNonNull", text);
        Assert.Contains("CanInstallNativeVTable", text);
        Assert.Contains("CanCallProcessDebugTensorRuntime", text);
        Assert.Contains("CanPromoteRealCallbackRuntime", text);
        Assert.Contains("ReasonNonNullAttachStillBlocked", text);
        Assert.Contains("ReasonNativeVTableStillBlocked", text);
        Assert.Contains("ReasonRuntimeProofStillBlocked", text);
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
