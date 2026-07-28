using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CalibratorMetadataDesignGateTests
{
    [Fact]
    public void KnownSurfaceGateReportsPresenceOnlyMetadataWithoutRuntimeProof()
    {
        TensorRtCalibratorMetadataDesignGateResult gate =
            TensorRtCalibratorMetadataDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt10);

        Assert.Equal("calibrator-metadata-design-gate", gate.EvidenceKind);
        Assert.Equal("calibrator-metadata", gate.DiagnosticsKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.IsRuntimeExecutionEvidence);
        Assert.False(gate.IsRuntimeExecutionProof);
        Assert.Equal(TensorRtApiLine.TensorRt10, gate.Line);
        Assert.True(gate.LineSupportsCalibrator);
        Assert.True(gate.PresenceProbeAvailable);
        Assert.True(gate.CopiedAlgorithmMetadataReady);
        Assert.True(gate.CopiedInterfaceInfoMetadataReady);
        Assert.False(gate.BatchCallbackOwnershipModeled);
        Assert.False(gate.CacheBufferOwnershipModeled);
        Assert.False(gate.CalibratorPointerExposed);
        Assert.False(gate.CalibratorPointerProduced);
        Assert.False(gate.BorrowedCalibratorPointerEscaped);
        Assert.False(gate.CallbackInvocationEnabled);
        Assert.False(gate.BatchBufferAccessEnabled);
        Assert.False(gate.CacheBufferAccessEnabled);
        Assert.True(gate.DirectCalibratorCallbackRowsDeferred);
        Assert.True(gate.DirectCalibratorCacheRowsDeferred);
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
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("getBatch callback buffer ownership", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("calibration cache", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("runtime execution proof", StringComparison.Ordinal));
        Assert.Contains("PresenceProbeAvailable=True", gate.Diagnostic);
        Assert.Contains("CallbackInvocationEnabled=False", gate.Diagnostic);
        Assert.Contains("BatchBufferAccessEnabled=False", gate.Diagnostic);
        Assert.Contains("CacheBufferAccessEnabled=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawCalibratorPointersOrCallbackControls()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtCalibratorMetadataDesignGate),
            typeof(TensorRtCalibratorMetadataDesignGateResult)
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
                string[] disabledCapabilityProperties =
                {
                    "CalibratorPointerExposed",
                    "CalibratorPointerProduced",
                    "BorrowedCalibratorPointerEscaped",
                    "CallbackInvocationEnabled",
                    "BatchBufferAccessEnabled",
                    "CacheBufferAccessEnabled",
                    "BatchCallbackOwnershipModeled",
                    "CacheBufferOwnershipModeled"
                };

                if (property.PropertyType == typeof(bool) &&
                    disabledCapabilityProperties.Contains(property.Name, StringComparer.Ordinal))
                {
                    object? instance = property.GetGetMethod()?.IsStatic == true ? null : Activator.CreateInstance(type);
                    object? value = property.GetValue(instance);
                    if (value is bool flag)
                    {
                        Assert.False(flag);
                    }
                }
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                Assert.NotEqual(typeof(IntPtr), method.ReturnType);
                Assert.NotEqual(typeof(UIntPtr), method.ReturnType);
                if (!method.IsSpecialName)
                {
                    Assert.DoesNotContain("GetBatch", method.Name, StringComparison.OrdinalIgnoreCase);
                    Assert.DoesNotContain("Cache", method.Name, StringComparison.OrdinalIgnoreCase);
                    Assert.DoesNotContain("Callback", method.Name, StringComparison.OrdinalIgnoreCase);
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
    public void DocsSmokeReadinessAndCoverageKeepDirectCalibratorRowsDeferred()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtCalibratorMetadataDesignGate.cs");
        string builderConfig = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string releaseEvidence = ReadSource("eng", "Export-ReleaseEvidenceBundle.ps1");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "calibrator-metadata-design-gate.md");
        string manualGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string trt8Deferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-tenth-batch-other-deferred.manifest.json");
        string trt10Deferred = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-third-batch-other-deferred.manifest.json");

        Assert.Contains("public static class TensorRtCalibratorMetadataDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtCalibratorMetadataDesignGateResult", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("IsRuntimeExecutionEvidence => false", gateSource);
        Assert.Contains("IsRuntimeExecutionProof => false", gateSource);
        Assert.Contains("CalibratorPointerExposed => false", gateSource);
        Assert.Contains("CallbackInvocationEnabled => false", gateSource);
        Assert.Contains("BatchBufferAccessEnabled => false", gateSource);
        Assert.Contains("CacheBufferAccessEnabled => false", gateSource);
        Assert.Contains("DirectCalibratorCallbackRowsDeferred => true", gateSource);
        Assert.Contains("DirectCalibratorCacheRowsDeferred => true", gateSource);
        Assert.Contains("CanPromoteWithoutRuntimeProof => false", gateSource);
        Assert.Contains("DeferredRowsStillRequired => true", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("public bool HasInt8CalibratorCompatibility", builderConfig);
        Assert.Contains("presence", builderConfig, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("does not transfer ownership", builderConfig, StringComparison.OrdinalIgnoreCase);

        Assert.Contains("calibrator-metadata-design-gate", smokeProgram);
        Assert.Contains("CalibratorMetadataDesignGate=", smokeProgram);
        Assert.Contains("PresenceProbeAvailable", smokeProgram);
        Assert.Contains("CalibratorPointerExposed", smokeProgram);
        Assert.Contains("CallbackInvocationEnabled", smokeProgram);
        Assert.Contains("RuntimeProofBlocked", smokeProgram);

        Assert.Contains("Calibrator Metadata Design Gate", designDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", designDoc);
        Assert.Contains("PresenceProbeAvailable=True", designDoc);
        Assert.Contains("CalibratorPointerExposed=False", designDoc);
        Assert.Contains("CallbackInvocationEnabled=False", designDoc);
        Assert.Contains("BatchBufferAccessEnabled=False", designDoc);
        Assert.Contains("CacheBufferAccessEnabled=False", designDoc);
        Assert.Contains("DirectCalibratorCallbackRowsDeferred=True", designDoc);
        Assert.Contains("RuntimeProofBlocked=True", designDoc);
        Assert.Contains("not proof", designDoc);
        Assert.Contains("calibrator-metadata-design-gate.md", docsIndex);
        Assert.Contains("calibrator-metadata-design-gate.md", docsToc);
        Assert.Contains("calibrator-metadata-design-gate", latest);
        Assert.Contains("calibrator-callback-metadata-design", manualGroups);
        Assert.Contains("已进入 `calibrator-metadata-design-gate`", manualGroups);

        Assert.Contains("New-CalibratorMetadataDesignGateEvidence", readiness);
        Assert.Contains("calibratorMetadataDesignGate", readiness);
        Assert.Contains("hasCalibratorMetadataDesignGate", readiness);
        Assert.Contains("source-smoke-docs-coverage", readiness);
        Assert.Contains("Calibrator metadata design gate missing evidence", readiness);
        Assert.Contains("calibrator-metadata-design-gate", releaseEvidence);

        Assert.Contains("\"IInt8Calibrator\",\"getAlgorithm\",\"IInt8Calibrator::getAlgorithm\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IInt8Calibrator\",\"getBatch\",\"IInt8Calibrator::getBatch\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IInt8Calibrator\",\"getBatchSize\",\"IInt8Calibrator::getBatchSize\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IInt8LegacyCalibrator\",\"getQuantile\",\"IInt8LegacyCalibrator::getQuantile\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IInt8LegacyCalibrator\",\"getRegressionCutoff\",\"IInt8LegacyCalibrator::getRegressionCutoff\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("int8-calibrator-read-calibration-cache-deferred", comparison);
        Assert.Contains("int8-calibrator-write-calibration-cache-deferred", comparison);
        Assert.Contains("trt8-int8-calibrator-get-batch-deferred", trt8Deferred);
        Assert.Contains("trt8-int8-calibrator-read-calibration-cache-deferred", trt8Deferred);
        Assert.Contains("trt8-int8-legacy-calibrator-get-regression-cutoff-deferred", trt8Deferred);
        Assert.Contains("trt10-int8-calibrator-get-batch-deferred", trt10Deferred);
        Assert.Contains("trt10-int8-calibrator-read-calibration-cache-deferred", trt10Deferred);
        Assert.Contains("trt10-int8-legacy-calibrator-get-regression-cutoff-deferred", trt10Deferred);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
