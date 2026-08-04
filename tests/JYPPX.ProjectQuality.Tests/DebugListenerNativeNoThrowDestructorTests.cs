using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeNoThrowDestructorTests
{
    [Fact]
    public void NativeNoThrowDestructorPromotesDestructorGateWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-nothrow-destructor",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult storage =
            TensorRtDebugListenerNativeOwnerNonCopyableStorage.Evaluate(disposed);
        TensorRtDebugListenerNativeNoThrowDestructorResult destructor =
            TensorRtDebugListenerNativeNoThrowDestructor.Evaluate(disposed, storage);

        Assert.Equal("debug-listener-native-nothrow-destructor", destructor.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", destructor.CallbackKind);
        Assert.Equal("destructor-gate", destructor.RuntimeEvidenceKind);
        Assert.False(destructor.RealCallbackRuntime);
        Assert.False(destructor.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, destructor.Line);
        Assert.True(destructor.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, destructor.LastStatus);
        Assert.True(destructor.NativeOwnerNonCopyableStorageReady);
        Assert.True(destructor.NativeOwnerNonCopyableReady);
        Assert.True(destructor.NativeOwnerCopyBlocked);
        Assert.True(destructor.NativeOwnerMoveBlocked);
        Assert.False(destructor.NativeOwnerAddressExposed);
        Assert.False(destructor.NativeOwnerPointerProduced);
        Assert.True(destructor.DestructorNoThrowScaffoldReady);
        Assert.True(destructor.DestructorExceptionEscapeBlocked);
        Assert.False(destructor.DestructorAddressExposed);
        Assert.False(destructor.DestructorPointerProduced);
        Assert.False(destructor.NativeAttachEntryLocated);
        Assert.True(destructor.NativeDetachEntryLocated);
        Assert.True(destructor.NoThrowNativeDestructorReady);
        Assert.False(destructor.NativeOwnerLifecycleReady);
        Assert.False(destructor.ProcessDebugTensorRuntimeReady);
        Assert.False(destructor.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(destructor.CanImplementNativeAttach);
        Assert.False(destructor.CanAttemptRuntimeProof);
        Assert.True(destructor.RuntimeProofBlocked);
        Assert.True(destructor.DeferredRowsStillRequired);
        Assert.Equal("destructor-gate-ready", destructor.Status);
        Assert.Contains(destructor.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.DoesNotContain(destructor.BlockedPrerequisites, item => item.Contains("no-throw destructor has not been promoted", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(destructor.BlockedPrerequisites, item => item.Contains("native DebugListener owner lifecycle is not complete", StringComparison.Ordinal));
        Assert.Contains(destructor.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=destructor-gate", destructor.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableStorageReady=True", destructor.Diagnostic);
        Assert.Contains("DestructorNoThrowScaffoldReady=True", destructor.Diagnostic);
        Assert.Contains("DestructorExceptionEscapeBlocked=True", destructor.Diagnostic);
        Assert.Contains("DestructorAddressExposed=False", destructor.Diagnostic);
        Assert.Contains("DestructorPointerProduced=False", destructor.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", destructor.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleReady=False", destructor.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", destructor.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", destructor.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeNoThrowDestructorAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-nothrow-destructor-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.DestructorNoThrowScaffoldReady);
        Assert.True(precheck.DestructorExceptionEscapeBlocked);
        Assert.False(precheck.DestructorAddressExposed);
        Assert.False(precheck.DestructorPointerProduced);
        Assert.True(precheck.NoThrowNativeDestructorReady);
        Assert.False(precheck.NativeOwnerLifecycleReady);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorNoThrowScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorExceptionEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("DestructorAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("DestructorPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeNoThrowDestructorSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeNoThrowDestructor),
            typeof(TensorRtDebugListenerNativeNoThrowDestructorResult)
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
    public void SourceSmokeAndDocsKeepNativeNoThrowDestructorSeparateFromRuntimeProof()
    {
        string destructorSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeNoThrowDestructor.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string nativeDestructor = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_nothrow_destructor.inc");
        string trt8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeNoThrowDestructor", destructorSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeNoThrowDestructorResult", destructorSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-nothrow-destructor\"", destructorSource);
        Assert.Contains("RuntimeEvidenceKind => \"destructor-gate\"", destructorSource);
        Assert.Contains("RealCallbackRuntime => false", destructorSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", destructorSource);
        Assert.Contains("NativeOwnerNonCopyableStorageReady", destructorSource);
        Assert.Contains("DestructorNoThrowScaffoldReady", destructorSource);
        Assert.Contains("DestructorExceptionEscapeBlocked", destructorSource);
        Assert.Contains("DestructorAddressExposed", destructorSource);
        Assert.Contains("DestructorPointerProduced", destructorSource);
        Assert.DoesNotContain("public IntPtr", destructorSource);
        Assert.DoesNotContain("public nint", destructorSource);

        Assert.Contains("TensorRtDebugListenerNativeNoThrowDestructor", precheckSource);
        Assert.Contains("NativeNoThrowDestructorGateReady", precheckSource);
        Assert.Contains("DestructorNoThrowScaffoldReady", precheckSource);
        Assert.Contains("DestructorExceptionEscapeBlocked", precheckSource);

        Assert.Contains("debug-listener-native-nothrow-destructor", smokeProgram);
        Assert.Contains("DebugListenerNativeNoThrowDestructor=", smokeProgram);
        Assert.Contains("NativeNoThrowDestructorGateReady", smokeProgram);

        Assert.Contains("struct DebugListenerNativeNoThrowDestructor final", nativeDestructor);
        Assert.Contains("DebugListenerNativeNoThrowDestructor(const DebugListenerNativeNoThrowDestructor&) = delete", nativeDestructor);
        Assert.Contains("operator=(const DebugListenerNativeNoThrowDestructor&) = delete", nativeDestructor);
        Assert.Contains("DebugListenerNativeNoThrowDestructor(DebugListenerNativeNoThrowDestructor&&) = delete", nativeDestructor);
        Assert.Contains("operator=(DebugListenerNativeNoThrowDestructor&&) = delete", nativeDestructor);
        Assert.Contains("~DebugListenerNativeNoThrowDestructor() noexcept = default", nativeDestructor);
        Assert.Contains("std::is_nothrow_destructible", nativeDestructor);
        Assert.Contains("debug_listener_native_nothrow_destructor.inc", trt8);
        Assert.Contains("debug_listener_native_nothrow_destructor.inc", trt10);
        Assert.Contains("debug_listener_native_nothrow_destructor.inc", trt11);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
