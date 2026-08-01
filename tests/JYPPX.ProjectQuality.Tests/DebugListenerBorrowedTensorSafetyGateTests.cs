using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerBorrowedTensorSafetyGateTests
{
    [Fact]
    public void GateCopiesBorrowedTensorMetadataWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-borrowed-tensor-safety-gate",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);
        Assert.Equal(1, diagnostic.ProcessDebugTensorCount);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(disposed);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult gate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(disposed, attachDetachGate);

        Assert.Equal("debug-listener-borrowed-tensor-safety-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-borrowed-tensor", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.Equal("quality_debug_tensor", gate.TensorName);
        Assert.Equal(TensorRtDataType.Float, gate.DataType);
        Assert.Equal(TensorRtTensorLocation.Device, gate.Location);
        Assert.Equal(4, gate.ShapeRank);
        Assert.True(gate.IsInput);
        Assert.False(gate.IsOutput);
        Assert.False(gate.IsShapeTensor);
        Assert.True(gate.IsExecutionTensor);
        Assert.Equal(1, gate.ProcessDebugTensorCount);
        Assert.True(gate.AttachDetachDesignGateReady);
        Assert.True(gate.OwnerDesignReady);
        Assert.True(gate.DebugTensorMetadataCopied);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(gate.BorrowedDebugTensorLifetimeReady);
        Assert.False(gate.BorrowedDebugTensorDataLifetimeReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(gate.SafetyGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.Equal("safety-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 4);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("borrowed debug tensor pointer lifetime", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("borrowed debug tensor data buffer lifetime", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("IDebugListener::processDebugTensor runtime callback", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("real-callback-runtime debug listener borrowed tensor", StringComparison.Ordinal));
        Assert.Contains("SafetyGateReady=True", gate.Diagnostic);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked=True", gate.Diagnostic);
        Assert.Contains("BorrowedDebugTensorLifetimeReady=False", gate.Diagnostic);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady=False", gate.Diagnostic);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerBorrowedTensorSafetyGate),
            typeof(TensorRtDebugListenerBorrowedTensorSafetyGateResult)
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
    public void ReadinessSmokeAndDocsKeepGateSeparateFromRuntimeProof()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerBorrowedTensorSafetyGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-borrowed-tensor-safety-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string attachDetachDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-attach-detach-design-gate.md");
        string callbackOwnerDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-callback-owner-design.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerBorrowedTensorSafetyGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerBorrowedTensorSafetyGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"debug-listener-borrowed-tensor-safety-gate\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("DebugTensorMetadataCopied", gateSource);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked", gateSource);
        Assert.Contains("BorrowedDebugTensorLifetimeReady", gateSource);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady", gateSource);
        Assert.Contains("ProcessDebugTensorRuntimeReady", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtDebugListenerBorrowedTensorSafetyGate", precheckSource);
        Assert.Contains("BorrowedTensorSafetyGateReady", precheckSource);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked", precheckSource);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady", precheckSource);
        Assert.Contains("ProcessDebugTensorRuntimeReady", precheckSource);

        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", smokeProgram);
        Assert.Contains("DebugListenerBorrowedTensorSafetyGate=", smokeProgram);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked", smokeProgram);
        Assert.Contains("ProcessDebugTensorRuntimeReady", smokeProgram);

        Assert.Contains("debugListenerBorrowedTensorSafetyGate", readiness);
        Assert.Contains("New-DebugListenerBorrowedTensorSafetyGateEvidence", readiness);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", readiness);
        Assert.Contains("safety-gate-ready", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", packageConsumer);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerBorrowedTensorSafetyGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerBorrowedTensorSafetyGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked=True", gateDoc);
        Assert.Contains("BorrowedDebugTensorLifetimeReady=False", gateDoc);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady=False", gateDoc);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", precheckDoc);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", attachDetachDoc);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", callbackOwnerDoc);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", trampolineGate);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", schema);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", latest);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", smokeReadme);

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
