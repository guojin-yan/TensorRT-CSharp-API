using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DimensionExpressionSnapshotDesignGateTests
{
    [Fact]
    public void KnownSurfaceGateReportsPointerFreeDesignWithoutRuntimeProof()
    {
        TensorRtDimensionExpressionSnapshotDesignGateResult gate =
            TensorRtDimensionExpressionSnapshotDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("dimension-expression-snapshot-design-gate", gate.EvidenceKind);
        Assert.Equal("dimension-expression-snapshot", gate.DiagnosticsKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.IsRuntimeExecutionEvidence);
        Assert.False(gate.IsRuntimeExecutionProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.LineSupportsDimensionExpression);
        Assert.True(gate.LineSupportsSizeTensor);
        Assert.True(gate.SnapshotTypeReady);
        Assert.True(gate.ConstantSnapshotCopyReady);
        Assert.True(gate.SizeTensorMetadataCopyReady);
        Assert.False(gate.OwnerLifetimeKnown);
        Assert.False(gate.ExprBuilderOwnershipModeled);
        Assert.False(gate.PluginShapeCallbackLifetimeModeled);
        Assert.False(gate.ExpressionPointerExposed);
        Assert.False(gate.ExpressionPointerProduced);
        Assert.False(gate.BorrowedExpressionPointerEscaped);
        Assert.False(gate.ExprBuilderPointerExposed);
        Assert.False(gate.ExprBuilderCreationEnabled);
        Assert.False(gate.ExpressionNodePublicOwnershipControl);
        Assert.True(gate.DirectDimensionExpressionRowsDeferred);
        Assert.True(gate.DirectExpressionBuilderRowsDeferred);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.True(gate.CopiedMetadataShapeReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.CanPromoteWithoutDesignGate);
        Assert.False(gate.CanPromoteWithoutRuntimeProof);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanPromoteRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("design-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 5);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("owner object lifetime", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("IExprBuilder expression node ownership", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("plugin shape callback", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("runtime execution proof", StringComparison.Ordinal));
        Assert.Contains("SnapshotTypeReady=True", gate.Diagnostic);
        Assert.Contains("OwnerLifetimeKnown=False", gate.Diagnostic);
        Assert.Contains("PointerFreeSurfaceReady=True", gate.Diagnostic);
        Assert.Contains("ExprBuilderCreationEnabled=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void TensorRt8KnownSurfaceSkipsSizeTensorButKeepsSameDesignBoundary()
    {
        TensorRtDimensionExpressionSnapshotDesignGateResult gate =
            TensorRtDimensionExpressionSnapshotDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt8);

        Assert.True(gate.LineSupportsDimensionExpression);
        Assert.False(gate.LineSupportsSizeTensor);
        Assert.True(gate.SizeTensorMetadataCopyReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.OwnerLifetimeKnown);
        Assert.False(gate.ExprBuilderCreationEnabled);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawDimensionExpressionPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDimensionExpressionSnapshotDesignGate),
            typeof(TensorRtDimensionExpressionSnapshotDesignGateResult)
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
                    Assert.DoesNotContain("Create", method.Name, StringComparison.OrdinalIgnoreCase);
                    Assert.DoesNotContain("BuilderPointer", method.Name, StringComparison.OrdinalIgnoreCase);
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
    public void DocsSmokeReadinessAndCoverageKeepDirectRowsDeferred()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDimensionExpressionSnapshotDesignGate.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "dimension-expression-snapshot-design-gate.md");
        string manualGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string releaseEvidence = ReadSource("eng", "Export-ReleaseEvidenceBundle.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string trt8Dimension = ReadTensorRtManifest("v8", "trt8-cross-version-tenth-batch-other-deferred.manifest.json");
        string trt8ExprBuilder = ReadTensorRtManifest("v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string trt10Dimension = ReadTensorRtManifest("v10", "trt10-cross-version-third-batch-other-deferred.manifest.json");
        string trt10ExprBuilder = ReadTensorRtManifest("v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");
        string trt11Dimension = ReadTensorRtManifest("v11", "trt11-forty-fifth-batch-callback-deferred.manifest.json");
        string trt11ExprBuilder = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("public static class TensorRtDimensionExpressionSnapshotDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDimensionExpressionSnapshotDesignGateResult", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("IsRuntimeExecutionEvidence => false", gateSource);
        Assert.Contains("IsRuntimeExecutionProof => false", gateSource);
        Assert.Contains("OwnerLifetimeKnown", gateSource);
        Assert.Contains("ExpressionPointerExposed => false", gateSource);
        Assert.Contains("BorrowedExpressionPointerEscaped => false", gateSource);
        Assert.Contains("ExprBuilderCreationEnabled => false", gateSource);
        Assert.Contains("DirectDimensionExpressionRowsDeferred => true", gateSource);
        Assert.Contains("DirectExpressionBuilderRowsDeferred => true", gateSource);
        Assert.Contains("CanPromoteWithoutRuntimeProof => false", gateSource);
        Assert.Contains("DeferredRowsStillRequired => true", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("dimension-expression-snapshot-design-gate", smokeProgram);
        Assert.Contains("DimensionExpressionSnapshotDesignGate=", smokeProgram);
        Assert.Contains("OwnerLifetimeKnown", smokeProgram);
        Assert.Contains("ExprBuilderCreationEnabled", smokeProgram);
        Assert.Contains("BorrowedExpressionPointerEscaped", smokeProgram);
        Assert.Contains("RuntimeProofBlocked", smokeProgram);

        Assert.Contains("Dimension Expression Snapshot Design Gate", designDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", designDoc);
        Assert.Contains("SnapshotTypeReady=True", designDoc);
        Assert.Contains("OwnerLifetimeKnown=False", designDoc);
        Assert.Contains("ExpressionPointerExposed=False", designDoc);
        Assert.Contains("BorrowedExpressionPointerEscaped=False", designDoc);
        Assert.Contains("ExprBuilderCreationEnabled=False", designDoc);
        Assert.Contains("DirectDimensionExpressionRowsDeferred=True", designDoc);
        Assert.Contains("DirectExpressionBuilderRowsDeferred=True", designDoc);
        Assert.Contains("RuntimeProofBlocked=True", designDoc);
        Assert.Contains("not proof", designDoc);
        Assert.Contains("dimension-expression-snapshot-design-gate.md", docsIndex);
        Assert.Contains("dimension-expression-snapshot-design-gate.md", docsToc);
        Assert.Contains("dimension-expression-snapshot-design", manualGroups);
        Assert.Contains("已进入 design gate", manualGroups);
        Assert.Contains("dimension-expression-snapshot-design-gate", latest);

        Assert.Contains("New-DimensionExpressionSnapshotDesignGateEvidence", readiness);
        Assert.Contains("dimensionExpressionSnapshotDesignGate", readiness);
        Assert.Contains("hasDimensionExpressionSnapshotDesignGate", readiness);
        Assert.Contains("source-smoke-docs-coverage", readiness);
        Assert.Contains("dimension expression snapshot design gate:", readiness);
        Assert.Contains("Dimension expression snapshot design gate missing evidence", readiness);
        Assert.Contains("errorRecorderDiagnosticsDesignGate", readiness);
        Assert.Contains("New-ErrorRecorderDiagnosticsDesignGateEvidence", readiness);
        Assert.Contains("error-recorder-diagnostics-design-gate", releaseEvidence);
        Assert.Contains("dimension-expression-snapshot-design-gate", releaseEvidence);
        Assert.Contains("Design gate evidence keeps IDimensionExpr/IExprBuilder deferred", releaseEvidence);

        Assert.Contains("\"IDimensionExpr\",\"getConstantValue\",\"IDimensionExpr::getConstantValue\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDimensionExpr\",\"isConstant\",\"IDimensionExpr::isConstant\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDimensionExpr\",\"isSizeTensor\",\"IDimensionExpr::isSizeTensor\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IExprBuilder\",\"constant\",\"IExprBuilder::constant\",\"builder\",\"deferred-only\"", comparison);
        Assert.Contains("\"IExprBuilder\",\"declareSizeTensor\",\"IExprBuilder::declareSizeTensor\",\"builder\",\"deferred-only\"", comparison);
        Assert.Contains("\"IExprBuilder\",\"operation\",\"IExprBuilder::operation\",\"builder\",\"deferred-only\"", comparison);

        Assert.Contains("trt8-dimension-expr-get-constant-value-deferred", trt8Dimension);
        Assert.Contains("trt8-dimension-expr-is-constant-deferred", trt8Dimension);
        Assert.Contains("jyppx_trt8_expr_builder_constant_deferred", trt8ExprBuilder);
        Assert.Contains("jyppx_trt8_expr_builder_operation_deferred", trt8ExprBuilder);
        Assert.Contains("trt10-dimension-expr-get-constant-value-deferred", trt10Dimension);
        Assert.Contains("trt10-dimension-expr-is-constant-deferred", trt10Dimension);
        Assert.Contains("trt10-dimension-expr-is-size-tensor-deferred", trt10Dimension);
        Assert.Contains("jyppx_trt10_expr_builder_constant_deferred", trt10ExprBuilder);
        Assert.Contains("jyppx_trt10_expr_builder_declare_size_tensor_deferred", trt10ExprBuilder);
        Assert.Contains("jyppx_trt10_expr_builder_operation_deferred", trt10ExprBuilder);
        Assert.Contains("jyppx_trt11_dimension_expr_get_constant_value_deferred", trt11Dimension);
        Assert.Contains("jyppx_trt11_dimension_expr_is_constant_deferred", trt11Dimension);
        Assert.Contains("jyppx_trt11_dimension_expr_is_size_tensor_deferred", trt11Dimension);
        Assert.Contains("jyppx_trt11_expr_builder_constant_deferred", trt11ExprBuilder);
        Assert.Contains("jyppx_trt11_expr_builder_declare_size_tensor_deferred", trt11ExprBuilder);
        Assert.Contains("jyppx_trt11_expr_builder_operation_deferred", trt11ExprBuilder);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
