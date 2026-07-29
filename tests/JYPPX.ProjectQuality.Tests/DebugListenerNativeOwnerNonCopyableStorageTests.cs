using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeOwnerNonCopyableStorageTests
{
    [Fact]
    public void NativeOwnerNonCopyableStoragePromotesStorageGateWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-noncopyable-storage",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeOwnerStableIdentityResult identity =
            TensorRtDebugListenerNativeOwnerStableIdentity.Evaluate(disposed);
        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult storage =
            TensorRtDebugListenerNativeOwnerNonCopyableStorage.Evaluate(disposed, identity);

        Assert.Equal("debug-listener-native-owner-noncopyable-storage", storage.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", storage.CallbackKind);
        Assert.Equal("storage-gate", storage.RuntimeEvidenceKind);
        Assert.False(storage.RealCallbackRuntime);
        Assert.False(storage.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, storage.Line);
        Assert.True(storage.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, storage.LastStatus);
        Assert.True(storage.NativeOwnerStableIdentityReady);
        Assert.True(storage.OwnerIdentityDiagnosticsReady);
        Assert.True(storage.OwnerIdentityPointerFree);
        Assert.True(storage.NativeOwnerNonCopyableReady);
        Assert.True(storage.NativeOwnerCopyBlocked);
        Assert.True(storage.NativeOwnerMoveBlocked);
        Assert.False(storage.NativeOwnerAddressExposed);
        Assert.False(storage.NativeOwnerPointerProduced);
        Assert.False(storage.NativeAttachEntryLocated);
        Assert.True(storage.NativeDetachEntryLocated);
        Assert.False(storage.NoThrowNativeDestructorReady);
        Assert.False(storage.NativeOwnerLifecycleReady);
        Assert.False(storage.ProcessDebugTensorRuntimeReady);
        Assert.False(storage.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(storage.CanImplementNativeAttach);
        Assert.False(storage.CanAttemptRuntimeProof);
        Assert.True(storage.RuntimeProofBlocked);
        Assert.True(storage.DeferredRowsStillRequired);
        Assert.Equal("storage-gate-ready", storage.Status);
        Assert.Contains(storage.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.DoesNotContain(storage.BlockedPrerequisites, item => item.Contains("non-copyable storage has not been implemented", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(storage.BlockedPrerequisites, item => item.Contains("no-throw destructor", StringComparison.Ordinal));
        Assert.Contains(storage.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=storage-gate", storage.Diagnostic);
        Assert.Contains("NativeOwnerStableIdentityReady=True", storage.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=True", storage.Diagnostic);
        Assert.Contains("NativeOwnerCopyBlocked=True", storage.Diagnostic);
        Assert.Contains("NativeOwnerMoveBlocked=True", storage.Diagnostic);
        Assert.Contains("NativeOwnerAddressExposed=False", storage.Diagnostic);
        Assert.Contains("NativeOwnerPointerProduced=False", storage.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", storage.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", storage.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeOwnerNonCopyableStorageAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-noncopyable-storage-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeOwnerStableIdentityReady);
        Assert.True(precheck.OwnerIdentityDiagnosticsReady);
        Assert.True(precheck.OwnerIdentityPointerFree);
        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.True(precheck.NativeOwnerCopyBlocked);
        Assert.True(precheck.NativeOwnerMoveBlocked);
        Assert.False(precheck.NativeOwnerAddressExposed);
        Assert.False(precheck.NativeOwnerPointerProduced);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
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
    public void PublicNativeOwnerNonCopyableStorageSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeOwnerNonCopyableStorage),
            typeof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult)
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
    public void SourceSmokeAndDocsKeepNativeOwnerNonCopyableStorageSeparateFromRuntimeProof()
    {
        string storageSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string nativeStorage = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_owner_noncopyable_storage.inc");
        string trt8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeOwnerNonCopyableStorage", storageSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeOwnerNonCopyableStorageResult", storageSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-owner-noncopyable-storage\"", storageSource);
        Assert.Contains("RuntimeEvidenceKind => \"storage-gate\"", storageSource);
        Assert.Contains("RealCallbackRuntime => false", storageSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", storageSource);
        Assert.Contains("NativeOwnerStableIdentityReady", storageSource);
        Assert.Contains("NativeOwnerNonCopyableReady", storageSource);
        Assert.Contains("NativeOwnerCopyBlocked", storageSource);
        Assert.Contains("NativeOwnerMoveBlocked", storageSource);
        Assert.Contains("NativeOwnerAddressExposed", storageSource);
        Assert.Contains("NativeOwnerPointerProduced", storageSource);
        Assert.DoesNotContain("public IntPtr", storageSource);
        Assert.DoesNotContain("public nint", storageSource);

        Assert.Contains("TensorRtDebugListenerNativeOwnerNonCopyableStorage", precheckSource);
        Assert.Contains("NativeOwnerNonCopyableStorageReady", precheckSource);
        Assert.Contains("NativeOwnerCopyBlocked", precheckSource);
        Assert.Contains("NativeOwnerMoveBlocked", precheckSource);
        Assert.Contains("NativeOwnerAddressExposed", precheckSource);
        Assert.Contains("NativeOwnerPointerProduced", precheckSource);

        Assert.Contains("debug-listener-native-owner-noncopyable-storage", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerNonCopyableStorage=", smokeProgram);
        Assert.Contains("NativeOwnerNonCopyableStorageReady", smokeProgram);

        Assert.Contains("struct DebugListenerNativeOwnerNonCopyableStorage final", nativeStorage);
        Assert.Contains("DebugListenerNativeOwnerNonCopyableStorage(const DebugListenerNativeOwnerNonCopyableStorage&) = delete", nativeStorage);
        Assert.Contains("operator=(const DebugListenerNativeOwnerNonCopyableStorage&) = delete", nativeStorage);
        Assert.Contains("DebugListenerNativeOwnerNonCopyableStorage(DebugListenerNativeOwnerNonCopyableStorage&&) = delete", nativeStorage);
        Assert.Contains("operator=(DebugListenerNativeOwnerNonCopyableStorage&&) = delete", nativeStorage);
        Assert.Contains("~DebugListenerNativeOwnerNonCopyableStorage() noexcept = default", nativeStorage);
        Assert.Contains("std::is_nothrow_destructible", nativeStorage);
        Assert.Contains("debug_listener_native_owner_noncopyable_storage.inc", trt8);
        Assert.Contains("debug_listener_native_owner_noncopyable_storage.inc", trt10);
        Assert.Contains("debug_listener_native_owner_noncopyable_storage.inc", trt11);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
