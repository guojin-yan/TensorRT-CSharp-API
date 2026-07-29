using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeAttachNoThrowPreflightTests
{
    [Fact]
    public void PreflightCopiesNativeAttachNoThrowStateWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-attach-nothrow-preflight",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(disposed);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(disposed, attachDetachGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult preflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate);

        Assert.Equal("debug-listener-native-attach-nothrow-preflight", preflight.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", preflight.CallbackKind);
        Assert.Equal("preflight", preflight.RuntimeEvidenceKind);
        Assert.False(preflight.RealCallbackRuntime);
        Assert.False(preflight.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, preflight.Line);
        Assert.True(preflight.AttachVTableSafetyGateReady);
        Assert.False(preflight.NativeAttachEntryLocated);
        Assert.True(preflight.NativeDetachEntryLocated);
        Assert.False(preflight.StableNativeOwnerAddressDesignReady);
        Assert.True(preflight.ManagedCallbackKeepAliveDesignReady);
        Assert.False(preflight.NoThrowVTableDesignReady);
        Assert.False(preflight.ExceptionToStatusMappingDesignReady);
        Assert.True(preflight.BorrowedDebugTensorMetadataCopyDesignReady);
        Assert.True(preflight.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(preflight.NativeVTableDesignReady);
        Assert.False(preflight.BorrowedDebugTensorLifetimeRuntimeReady);
        Assert.False(preflight.BorrowedDebugTensorDataLifetimeRuntimeReady);
        Assert.False(preflight.ProcessDebugTensorRuntimeReady);
        Assert.False(preflight.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(preflight.PreflightReady);
        Assert.False(preflight.CanImplementNativeAttach);
        Assert.False(preflight.CanAttemptRuntimeProof);
        Assert.True(preflight.RuntimeProofBlocked);
        Assert.True(preflight.DeferredRowsStillRequired);
        Assert.Equal("preflight-ready", preflight.Status);
        Assert.True(preflight.BlockedPrerequisiteCount >= 8);
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("stable native DebugListener owner address", StringComparison.Ordinal));
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("exception-to-status", StringComparison.Ordinal));
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("IDebugListener::processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=preflight", preflight.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", preflight.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", preflight.Diagnostic);
        Assert.Contains("NoThrowVTableDesignReady=False", preflight.Diagnostic);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", preflight.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", preflight.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", preflight.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeAttachNoThrowPreflightAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-attach-nothrow-preflight-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(disposed);

        Assert.True(precheck.NativeAttachNoThrowPreflightReady);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
        Assert.False(precheck.StableNativeOwnerAddressDesignReady);
        Assert.True(precheck.ManagedCallbackKeepAliveDesignReady);
        Assert.False(precheck.NoThrowVTableDesignReady);
        Assert.False(precheck.ExceptionToStatusMappingDesignReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeAttachNoThrowPreflightReady=True", precheck.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", precheck.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", precheck.Diagnostic);
        Assert.Contains("NoThrowVTableDesignReady=False", precheck.Diagnostic);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", precheck.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicPreflightSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeAttachNoThrowPreflight),
            typeof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult)
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
    public void ReadinessSmokeAndDocsKeepPreflightSeparateFromRuntimeProof()
    {
        string preflightSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeAttachNoThrowPreflight.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string preflightDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-nothrow-preflight.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string attachVTableDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-attach-vtable-safety-gate.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeAttachNoThrowPreflight", preflightSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeAttachNoThrowPreflightResult", preflightSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-attach-nothrow-preflight\"", preflightSource);
        Assert.Contains("RuntimeEvidenceKind => \"preflight\"", preflightSource);
        Assert.Contains("RealCallbackRuntime => false", preflightSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", preflightSource);
        Assert.Contains("NativeAttachEntryLocated", preflightSource);
        Assert.Contains("NoThrowVTableDesignReady", preflightSource);
        Assert.Contains("ExceptionToStatusMappingDesignReady", preflightSource);
        Assert.Contains("CanImplementNativeAttach", preflightSource);
        Assert.DoesNotContain("public IntPtr", preflightSource);
        Assert.DoesNotContain("public nint", preflightSource);

        Assert.Contains("TensorRtDebugListenerNativeAttachNoThrowPreflight", precheckSource);
        Assert.Contains("NativeAttachNoThrowPreflightReady", precheckSource);
        Assert.Contains("CanImplementNativeAttach", precheckSource);

        Assert.Contains("debug-listener-native-attach-nothrow-preflight", smokeProgram);
        Assert.Contains("DebugListenerNativeAttachNoThrowPreflight=", smokeProgram);
        Assert.Contains("CanImplementNativeAttach", smokeProgram);

        Assert.Contains("debugListenerNativeAttachNoThrowPreflight", readiness);
        Assert.Contains("New-DebugListenerNativeAttachNoThrowPreflightEvidence", readiness);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", readiness);
        Assert.Contains("preflight-ready", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-attach-nothrow-preflight", packageConsumer);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachNoThrowPreflight", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachNoThrowPreflightResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-attach-nothrow-preflight", preflightDoc);
        Assert.Contains("RuntimeEvidenceKind=preflight", preflightDoc);
        Assert.Contains("NativeAttachEntryLocated=False", preflightDoc);
        Assert.Contains("NativeDetachEntryLocated=True", preflightDoc);
        Assert.Contains("NoThrowVTableDesignReady=False", preflightDoc);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", preflightDoc);
        Assert.Contains("CanImplementNativeAttach=False", preflightDoc);
        Assert.Contains("not proof", preflightDoc);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", precheckDoc);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", attachVTableDoc);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", trampolineGate);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", schema);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", latest);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", index);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight.md", toc);
        Assert.Contains("debugListenerNativeAttachNoThrowPreflight", runtimeSplitReadme);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", smokeReadme);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
