using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ErrorRecorderDiagnosticsDesignGateTests
{
    [Fact]
    public void KnownSurfaceGateReportsCopiedDiagnosticsWithoutRuntimeProof()
    {
        TensorRtErrorRecorderDiagnosticsDesignGateResult gate =
            TensorRtErrorRecorderDiagnosticsDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("error-recorder-diagnostics-design-gate", gate.EvidenceKind);
        Assert.Equal("error-recorder-diagnostics", gate.DiagnosticsKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.IsRuntimeExecutionEvidence);
        Assert.False(gate.IsRuntimeExecutionProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.LineSupportsErrorRecorder);
        Assert.True(gate.SnapshotTypeAvailable);
        Assert.True(gate.RuntimeSnapshotAvailable);
        Assert.True(gate.RefitterSnapshotAvailable);
        Assert.True(gate.PresenceControlsAvailable);
        Assert.True(gate.ClearControlsAvailable);
        Assert.False(gate.CopiedSnapshotObserved);
        Assert.False(gate.HasRecorder);
        Assert.Equal(0, gate.ErrorCount);
        Assert.False(gate.HasOverflowed);
        Assert.Equal(0, gate.CopiedRecordCount);
        Assert.True(gate.CopiedDiagnosticsReady);
        Assert.True(gate.SnapshotRecordCopyReady);
        Assert.False(gate.RecorderPointerExposed);
        Assert.False(gate.RecorderPointerProduced);
        Assert.False(gate.BorrowedRecorderPointerEscaped);
        Assert.False(gate.RefCountPublicOwnershipControl);
        Assert.False(gate.InterfaceInfoPublicOwnershipControl);
        Assert.True(gate.DirectRecorderOwnershipDeferred);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.CanPromoteWithoutDesignGate);
        Assert.False(gate.CanPromoteWithoutRuntimeProof);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanPromoteRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("design-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 3);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("ref-count ownership", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("interface-info ownership", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("runtime execution proof", StringComparison.Ordinal));
        Assert.Contains("CopiedDiagnosticsReady=True", gate.Diagnostic);
        Assert.Contains("PointerFreeSurfaceReady=True", gate.Diagnostic);
        Assert.Contains("RefCountPublicOwnershipControl=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawRecorderPointersOrOwnershipControls()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtErrorRecorderDiagnosticsDesignGate),
            typeof(TensorRtErrorRecorderDiagnosticsDesignGateResult),
            typeof(TensorRtErrorRecorderSnapshot),
            typeof(TensorRtErrorRecord)
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
                if (!method.IsSpecialName)
                {
                    Assert.DoesNotContain("IncRef", method.Name, StringComparison.OrdinalIgnoreCase);
                    Assert.DoesNotContain("DecRef", method.Name, StringComparison.OrdinalIgnoreCase);
                    Assert.DoesNotContain("RefCount", method.Name, StringComparison.OrdinalIgnoreCase);
                }

                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }
        }
    }

    [Fact]
    public void DocsSmokeAndCoverageKeepDirectRecorderRowsDeferred()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtErrorRecorderDiagnosticsDesignGate.cs");
        string snapshotSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtErrorRecorderSnapshot.cs");
        string runtimeWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.Trt11Controls.cs");
        string refitterWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Refit", "TensorRtRefitter.Trt11Controls.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "error-recorder-diagnostics-design-gate.md");
        string snapshotDoc = ReadSource("docs", "articles", "zh-cn", "error-recorder-snapshot-guide.md");
        string manualGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string trt8Deferred = ReadTensorRtManifest("v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string trt10Deferred = ReadTensorRtManifest("v10", "trt10-cross-version-sixth-batch-diagnostics-refitter-deferred.manifest.json");
        string trt11Deferred = ReadTensorRtManifest("v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");

        Assert.Contains("public static class TensorRtErrorRecorderDiagnosticsDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtErrorRecorderDiagnosticsDesignGateResult", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("IsRuntimeExecutionEvidence => false", gateSource);
        Assert.Contains("IsRuntimeExecutionProof => false", gateSource);
        Assert.Contains("RecorderPointerExposed => false", gateSource);
        Assert.Contains("RefCountPublicOwnershipControl => false", gateSource);
        Assert.Contains("DirectRecorderOwnershipDeferred => true", gateSource);
        Assert.Contains("CanPromoteWithoutRuntimeProof => false", gateSource);
        Assert.Contains("DeferredRowsStillRequired => true", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("public sealed class TensorRtErrorRecorderSnapshot", snapshotSource);
        Assert.Contains("IReadOnlyList<TensorRtErrorRecord>", snapshotSource);
        Assert.DoesNotContain("public IntPtr", snapshotSource);
        Assert.DoesNotContain("public nint", snapshotSource);
        Assert.Contains("does not expose, retain, increment, decrement, or destroy", runtimeWrapper);
        Assert.Contains("does not expose, retain, increment, decrement, or destroy", refitterWrapper);

        Assert.Contains("error-recorder-diagnostics-design-gate", smokeProgram);
        Assert.Contains("ErrorRecorderDiagnosticsDesignGate=", smokeProgram);
        Assert.Contains("CopiedDiagnosticsReady", smokeProgram);
        Assert.Contains("RecorderPointerExposed", smokeProgram);
        Assert.Contains("RefCountPublicOwnershipControl", smokeProgram);
        Assert.Contains("RuntimeProofBlocked", smokeProgram);

        Assert.Contains("ErrorRecorder Diagnostics Design Gate", designDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", designDoc);
        Assert.Contains("CopiedDiagnosticsReady=True", designDoc);
        Assert.Contains("PointerFreeSurfaceReady=True", designDoc);
        Assert.Contains("RecorderPointerExposed=False", designDoc);
        Assert.Contains("RefCountPublicOwnershipControl=False", designDoc);
        Assert.Contains("RuntimeProofBlocked=True", designDoc);
        Assert.Contains("not proof", designDoc);
        Assert.Contains("error-recorder-diagnostics-design-gate.md", docsIndex);
        Assert.Contains("error-recorder-diagnostics-design-gate.md", docsToc);
        Assert.Contains("error-recorder-diagnostics-design-gate", snapshotDoc);
        Assert.Contains("error-recorder-diagnostics-design", manualGroups);
        Assert.Contains("已进入 design gate", manualGroups);
        Assert.Contains("error-recorder-diagnostics-design-gate", latest);

        Assert.Contains("\"IErrorRecorder\",\"getNbErrors\",\"IErrorRecorder::getNbErrors\",\"diagnostics\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IErrorRecorder\",\"getErrorCode\",\"IErrorRecorder::getErrorCode\",\"diagnostics\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IErrorRecorder\",\"getErrorDesc\",\"IErrorRecorder::getErrorDesc\",\"diagnostics\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IErrorRecorder\",\"getInterfaceInfo\",\"IErrorRecorder::getInterfaceInfo\",\"diagnostics\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("engine-get-error-recorder-snapshot-info", comparison);
        Assert.Contains("execution-context-get-error-recorder-snapshot-info", comparison);
        Assert.Contains("runtime-get-error-recorder-snapshot-info", comparison);
        Assert.Contains("refitter-get-error-recorder-snapshot-info", comparison);
        Assert.Contains("error-recorder-get-nb-errors-deferred", comparison);
        Assert.Contains("error-recorder-get-error-code-deferred", comparison);
        Assert.Contains("error-recorder-get-error-desc-deferred", comparison);
        Assert.Contains("error-recorder-get-interface-info-deferred", comparison);

        Assert.Contains("trt8-error-recorder-get-nb-errors-deferred", trt8Deferred);
        Assert.Contains("trt8-error-recorder-get-error-code-deferred", trt8Deferred);
        Assert.Contains("trt8-error-recorder-get-error-desc-deferred", trt8Deferred);
        Assert.Contains("trt10-error-recorder-get-nb-errors-deferred", trt10Deferred);
        Assert.Contains("trt10-error-recorder-get-error-code-deferred", trt10Deferred);
        Assert.Contains("trt10-error-recorder-get-error-desc-deferred", trt10Deferred);
        Assert.Contains("trt10-error-recorder-get-interface-info-deferred", trt10Deferred);
        Assert.Contains("trt11-error-recorder-get-nb-errors-deferred", trt11Deferred);
        Assert.Contains("trt11-error-recorder-get-error-code-deferred", trt11Deferred);
        Assert.Contains("trt11-error-recorder-get-error-desc-deferred", trt11Deferred);
        Assert.Contains("trt11-error-recorder-get-interface-info-deferred", trt11Deferred);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
