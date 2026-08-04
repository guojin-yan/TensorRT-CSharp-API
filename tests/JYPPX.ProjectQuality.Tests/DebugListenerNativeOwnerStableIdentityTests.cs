using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeOwnerStableIdentityTests
{
    [Fact]
    public void NativeOwnerStableIdentityCopiesPointerFreeIdentityEvidenceWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-stable-identity",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeOwnerStableIdentityResult identity =
            TensorRtDebugListenerNativeOwnerStableIdentity.Evaluate(disposed);

        Assert.Equal("debug-listener-native-owner-stable-identity", identity.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", identity.CallbackKind);
        Assert.Equal("identity-gate", identity.RuntimeEvidenceKind);
        Assert.False(identity.RealCallbackRuntime);
        Assert.False(identity.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, identity.Line);
        Assert.True(identity.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, identity.LastStatus);
        Assert.False(string.IsNullOrWhiteSpace(identity.LastDiagnostic));
        Assert.False(string.IsNullOrWhiteSpace(identity.ReleaseDiagnostic));
        Assert.True(identity.NativeAttachEntryRuntimeScaffoldReady);
        Assert.True(identity.StableNativeOwnerIdentityReady);
        Assert.True(identity.OwnerIdentityDiagnosticsReady);
        Assert.True(identity.OwnerIdentityPointerFree);
        Assert.False(identity.NativeAttachEntryLocated);
        Assert.True(identity.NativeDetachEntryLocated);
        Assert.False(identity.NativeOwnerNonCopyableReady);
        Assert.False(identity.NoThrowNativeDestructorReady);
        Assert.False(identity.NativeOwnerLifecycleReady);
        Assert.False(identity.ProcessDebugTensorRuntimeReady);
        Assert.False(identity.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(identity.CanImplementNativeAttach);
        Assert.False(identity.CanAttemptRuntimeProof);
        Assert.True(identity.RuntimeProofBlocked);
        Assert.True(identity.DeferredRowsStillRequired);
        Assert.Equal("identity-gate-ready", identity.Status);
        Assert.True(identity.BlockedPrerequisiteCount >= 4);
        Assert.Contains(identity.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(identity.BlockedPrerequisites, item => item.Contains("non-copyable storage", StringComparison.Ordinal));
        Assert.Contains(identity.BlockedPrerequisites, item => item.Contains("no-throw destructor", StringComparison.Ordinal));
        Assert.Contains(identity.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=identity-gate", identity.Diagnostic);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady=True", identity.Diagnostic);
        Assert.Contains("StableNativeOwnerIdentityReady=True", identity.Diagnostic);
        Assert.Contains("OwnerIdentityDiagnosticsReady=True", identity.Diagnostic);
        Assert.Contains("OwnerIdentityPointerFree=True", identity.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", identity.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=False", identity.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=False", identity.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", identity.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", identity.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeOwnerStableIdentityAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-stable-identity-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeAttachEntryRuntimeScaffoldReady);
        Assert.True(precheck.NativeOwnerStableIdentityReady);
        Assert.True(precheck.OwnerIdentityDiagnosticsReady);
        Assert.True(precheck.OwnerIdentityPointerFree);
        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerCopyBlocked);
        Assert.True(precheck.NativeOwnerMoveBlocked);
        Assert.False(precheck.NativeOwnerAddressExposed);
        Assert.False(precheck.NativeOwnerPointerProduced);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.DestructorNoThrowScaffoldReady);
        Assert.True(precheck.DestructorExceptionEscapeBlocked);
        Assert.False(precheck.DestructorAddressExposed);
        Assert.False(precheck.DestructorPointerProduced);
        Assert.True(precheck.NoThrowNativeDestructorReady);
        Assert.False(precheck.NativeOwnerLifecycleReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeOwnerStableIdentityReady=True", precheck.Diagnostic);
        Assert.Contains("OwnerIdentityDiagnosticsReady=True", precheck.Diagnostic);
        Assert.Contains("OwnerIdentityPointerFree=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableStorageReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerCopyBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerMoveBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", precheck.Diagnostic);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorNoThrowScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorExceptionEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("DestructorAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("DestructorPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", precheck.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeOwnerStableIdentitySurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeOwnerStableIdentity),
            typeof(TensorRtDebugListenerNativeOwnerStableIdentityResult)
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
    public void ReadinessSmokeAndDocsKeepNativeOwnerStableIdentitySeparateFromRuntimeProof()
    {
        string identitySource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerStableIdentity.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string identityDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-stable-identity.md");
        string scaffoldDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-entry-runtime-scaffold.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeOwnerStableIdentity", identitySource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeOwnerStableIdentityResult", identitySource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-owner-stable-identity\"", identitySource);
        Assert.Contains("RuntimeEvidenceKind => \"identity-gate\"", identitySource);
        Assert.Contains("RealCallbackRuntime => false", identitySource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", identitySource);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", identitySource);
        Assert.Contains("StableNativeOwnerIdentityReady", identitySource);
        Assert.Contains("OwnerIdentityDiagnosticsReady", identitySource);
        Assert.Contains("OwnerIdentityPointerFree", identitySource);
        Assert.Contains("NativeAttachEntryLocated", identitySource);
        Assert.Contains("NativeOwnerNonCopyableReady", identitySource);
        Assert.DoesNotContain("public IntPtr", identitySource);
        Assert.DoesNotContain("public nint", identitySource);

        Assert.Contains("TensorRtDebugListenerNativeOwnerStableIdentity", precheckSource);
        Assert.Contains("NativeOwnerStableIdentityReady", precheckSource);
        Assert.Contains("OwnerIdentityDiagnosticsReady", precheckSource);
        Assert.Contains("OwnerIdentityPointerFree", precheckSource);
        Assert.Contains("NativeOwnerStableIdentityReady &&", precheckSource);

        Assert.Contains("debug-listener-native-owner-stable-identity", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerStableIdentity=", smokeProgram);
        Assert.Contains("NativeOwnerStableIdentityReady", smokeProgram);

        Assert.Contains("debugListenerNativeOwnerStableIdentity", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerStableIdentityEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-stable-identity", readiness);
        Assert.Contains("identity-gate-ready", readiness);
        Assert.Contains("stableNativeOwnerIdentityReady = $true", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-owner-stable-identity", packageConsumer);
        Assert.Contains("debug-listener-native-owner-stable-identity", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerStableIdentity", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerStableIdentityResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-owner-stable-identity", identityDoc);
        Assert.Contains("RuntimeEvidenceKind=identity-gate", identityDoc);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady=True", identityDoc);
        Assert.Contains("StableNativeOwnerIdentityReady=True", identityDoc);
        Assert.Contains("OwnerIdentityDiagnosticsReady=True", identityDoc);
        Assert.Contains("OwnerIdentityPointerFree=True", identityDoc);
        Assert.Contains("NativeAttachEntryLocated=False", identityDoc);
        Assert.Contains("NativeOwnerNonCopyableReady=False", identityDoc);
        Assert.Contains("NoThrowNativeDestructorReady=False", identityDoc);
        Assert.Contains("not proof", identityDoc);
        Assert.Contains("debug-listener-native-owner-stable-identity", scaffoldDoc);
        Assert.Contains("debug-listener-native-owner-stable-identity", precheckDoc);
        Assert.Contains("debug-listener-native-owner-stable-identity", trampolineGate);
        Assert.Contains("debug-listener-native-owner-stable-identity", schema);
        Assert.Contains("debug-listener-native-owner-stable-identity", latest);
        Assert.Contains("debug-listener-native-owner-stable-identity", index);
        Assert.Contains("debug-listener-native-owner-stable-identity.md", toc);
        Assert.Contains("debug-listener-native-owner-stable-identity", smokeReadme);

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
