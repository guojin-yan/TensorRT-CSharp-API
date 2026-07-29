using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeAttachEntryRuntimeScaffoldTests
{
    [Fact]
    public void NativeAttachEntryRuntimeScaffoldCopiesShapeEvidenceWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-attach-entry-runtime-scaffold",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult scaffold =
            TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(disposed);

        Assert.Equal("debug-listener-native-attach-entry-runtime-scaffold", scaffold.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", scaffold.CallbackKind);
        Assert.Equal("scaffold", scaffold.RuntimeEvidenceKind);
        Assert.False(scaffold.RealCallbackRuntime);
        Assert.False(scaffold.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, scaffold.Line);
        Assert.True(scaffold.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, scaffold.LastStatus);
        Assert.True(scaffold.NativeOwnerLifecycleDryRunReady);
        Assert.False(scaffold.NativeAttachEntryLocated);
        Assert.True(scaffold.NativeDetachEntryLocated);
        Assert.True(scaffold.AttachEntryParameterShapeReady);
        Assert.True(scaffold.AttachEntryVersionGuardReady);
        Assert.True(scaffold.AttachEntryNoThrowBoundaryReady);
        Assert.True(scaffold.AttachEntryOwnershipDiagnosticsReady);
        Assert.False(scaffold.StableNativeOwnerIdentityReady);
        Assert.False(scaffold.NativeOwnerNonCopyableReady);
        Assert.False(scaffold.NoThrowNativeDestructorReady);
        Assert.False(scaffold.NativeOwnerLifecycleReady);
        Assert.True(scaffold.RuntimeScaffoldReady);
        Assert.False(scaffold.ProcessDebugTensorRuntimeReady);
        Assert.False(scaffold.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(scaffold.CanImplementNativeAttach);
        Assert.False(scaffold.CanAttemptRuntimeProof);
        Assert.True(scaffold.RuntimeProofBlocked);
        Assert.True(scaffold.DeferredRowsStillRequired);
        Assert.Equal("scaffold-ready", scaffold.Status);
        Assert.True(scaffold.BlockedPrerequisiteCount >= 4);
        Assert.Contains(scaffold.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(scaffold.BlockedPrerequisites, item => item.Contains("stable owner identity", StringComparison.Ordinal));
        Assert.Contains(scaffold.BlockedPrerequisites, item => item.Contains("non-copyable storage", StringComparison.Ordinal));
        Assert.Contains(scaffold.BlockedPrerequisites, item => item.Contains("no-throw destructor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=scaffold", scaffold.Diagnostic);
        Assert.Contains("RuntimeScaffoldReady=True", scaffold.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleDryRunReady=True", scaffold.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", scaffold.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", scaffold.Diagnostic);
        Assert.Contains("AttachEntryParameterShapeReady=True", scaffold.Diagnostic);
        Assert.Contains("AttachEntryVersionGuardReady=True", scaffold.Diagnostic);
        Assert.Contains("AttachEntryNoThrowBoundaryReady=True", scaffold.Diagnostic);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady=True", scaffold.Diagnostic);
        Assert.Contains("StableNativeOwnerIdentityReady=False", scaffold.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=False", scaffold.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=False", scaffold.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", scaffold.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", scaffold.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeAttachEntryRuntimeScaffoldAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-attach-entry-runtime-scaffold-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeOwnerLifecycleDryRunReady);
        Assert.True(precheck.NativeAttachEntryRuntimeScaffoldReady);
        Assert.True(precheck.AttachEntryParameterShapeReady);
        Assert.True(precheck.AttachEntryNoThrowBoundaryReady);
        Assert.True(precheck.AttachEntryOwnershipDiagnosticsReady);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
        Assert.False(precheck.LineSpecificAttachEntryDesignReady);
        Assert.False(precheck.AttachEntryNoThrowReady);
        Assert.False(precheck.AttachEntryVersionGuardReady);
        Assert.False(precheck.AttachEntryOwnershipReady);
        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.True(precheck.NativeOwnerCopyBlocked);
        Assert.True(precheck.NativeOwnerMoveBlocked);
        Assert.False(precheck.NativeOwnerAddressExposed);
        Assert.False(precheck.NativeOwnerPointerProduced);
        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.DestructorNoThrowScaffoldReady);
        Assert.True(precheck.DestructorExceptionEscapeBlocked);
        Assert.False(precheck.DestructorAddressExposed);
        Assert.False(precheck.DestructorPointerProduced);
        Assert.True(precheck.NoThrowNativeDestructorReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("AttachEntryParameterShapeReady=True", precheck.Diagnostic);
        Assert.Contains("AttachEntryNoThrowBoundaryReady=True", precheck.Diagnostic);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady=True", precheck.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableStorageReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerCopyBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerMoveBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorNoThrowScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorExceptionEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("DestructorAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("DestructorPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", precheck.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeAttachEntryRuntimeScaffoldSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffold),
            typeof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult)
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
    public void ReadinessSmokeAndDocsKeepNativeAttachEntryRuntimeScaffoldSeparateFromRuntimeProof()
    {
        string scaffoldSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string scaffoldDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-entry-runtime-scaffold.md");
        string dryRunDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-lifecycle-dry-run.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeAttachEntryRuntimeScaffold", scaffoldSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult", scaffoldSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-attach-entry-runtime-scaffold\"", scaffoldSource);
        Assert.Contains("RuntimeEvidenceKind => \"scaffold\"", scaffoldSource);
        Assert.Contains("RealCallbackRuntime => false", scaffoldSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", scaffoldSource);
        Assert.Contains("RuntimeScaffoldReady", scaffoldSource);
        Assert.Contains("AttachEntryParameterShapeReady", scaffoldSource);
        Assert.Contains("AttachEntryVersionGuardReady", scaffoldSource);
        Assert.Contains("AttachEntryNoThrowBoundaryReady", scaffoldSource);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady", scaffoldSource);
        Assert.DoesNotContain("public IntPtr", scaffoldSource);
        Assert.DoesNotContain("public nint", scaffoldSource);

        Assert.Contains("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold", precheckSource);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", precheckSource);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady &&", precheckSource);

        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", smokeProgram);
        Assert.Contains("DebugListenerNativeAttachEntryRuntimeScaffold=", smokeProgram);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", smokeProgram);

        Assert.Contains("debugListenerNativeAttachEntryRuntimeScaffold", readiness);
        Assert.Contains("New-DebugListenerNativeAttachEntryRuntimeScaffoldEvidence", readiness);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", readiness);
        Assert.Contains("scaffold-ready", readiness);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", packageConsumer);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", scaffoldDoc);
        Assert.Contains("RuntimeEvidenceKind=scaffold", scaffoldDoc);
        Assert.Contains("RuntimeScaffoldReady=True", scaffoldDoc);
        Assert.Contains("NativeOwnerLifecycleDryRunReady=True", scaffoldDoc);
        Assert.Contains("NativeAttachEntryLocated=False", scaffoldDoc);
        Assert.Contains("NativeDetachEntryLocated=True", scaffoldDoc);
        Assert.Contains("AttachEntryParameterShapeReady=True", scaffoldDoc);
        Assert.Contains("AttachEntryVersionGuardReady=True", scaffoldDoc);
        Assert.Contains("AttachEntryNoThrowBoundaryReady=True", scaffoldDoc);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady=True", scaffoldDoc);
        Assert.Contains("StableNativeOwnerIdentityReady=False", scaffoldDoc);
        Assert.Contains("NativeOwnerNonCopyableReady=False", scaffoldDoc);
        Assert.Contains("NoThrowNativeDestructorReady=False", scaffoldDoc);
        Assert.Contains("not proof", scaffoldDoc);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", dryRunDoc);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", precheckDoc);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", trampolineGate);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", schema);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", latest);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", index);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold.md", toc);
        Assert.Contains("debugListenerNativeAttachEntryRuntimeScaffold", runtimeSplitReadme);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", smokeReadme);

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
