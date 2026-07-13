using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PackageConsumerRuntimeProofPreflightMatrixTests
{
    private static readonly string[] RequiredRuntimeKeys =
    {
        "win-x64-trt8.6-cuda11.8-cudnn8.9",
        "win-x64-trt8.6-cuda12.1-cudnn8.9",
        "win-x64-trt10.11-cuda11.8-cudnn8.9",
        "win-x64-trt10.11-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda13.2-cudnn9.22"
    };

    private static readonly string[] RequiredForbiddenSubstitutes =
    {
        "readonly summary",
        "readonly diagnostics",
        "TensorRtExec report",
        "OnnxToEngine report",
        "YoloVision matrix",
        "bridge-only",
        "dependency probe",
        "local feed",
        "ProjectReference",
        "direct .nupkg",
        "build-only",
        "dry-run",
        "template",
        "blocked-by-cuda-driver"
    };

    private static readonly string[] RequiredSummaryMarkers =
    {
        "EngineDeploymentSummary=",
        "BuilderConfigDeploymentSummary=",
        "ExecutionContextDeploymentSummary=",
        "SerializationConfigSummary=",
        "RuntimeConfigSummary=",
        "GraphDiagnosticSummary=",
        "GraphExecDiagnosticSummary=",
        "MemoryRangeSummary="
    };

    [Fact]
    public void PreflightMatrixCoversWindowsRuntimeKeysWithoutPromotingProof()
    {
        JsonElement root = ReadMatrix();
        JsonElement boundary = root.GetProperty("proofBoundary");
        JsonElement[] manifestPackages = ReadRuntimeManifestPackages()
            .Where(static package => package.GetProperty("platform").GetString() == "windows")
            .ToArray();

        Assert.False(boundary.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPromoteAnyEntry").GetBoolean());
        Assert.Equal("JYPPX.TensorRT.CSharp.API", root.GetProperty("managedPackageId").GetString());

        JsonElement entries = root.GetProperty("entries");
        Assert.Equal(RequiredRuntimeKeys.Length, entries.GetArrayLength());
        Assert.Equal(RequiredRuntimeKeys.Length, manifestPackages.Length);

        foreach (string runtimeKey in RequiredRuntimeKeys)
        {
            JsonElement entry = FindEntry(entries, runtimeKey);
            JsonElement manifestPackage = FindManifestPackage(manifestPackages, runtimeKey);
            int expectedNativeAssetCount = CountExpectedNativeAssets(manifestPackage);

            Assert.Equal(runtimeKey, entry.GetProperty("runtimePackageKey").GetString());
            Assert.Equal("JYPPX.TensorRT.CSharp.API", entry.GetProperty("managedPackageId").GetString());
            Assert.Equal(manifestPackage.GetProperty("packageId").GetString(), entry.GetProperty("runtimePackageId").GetString());
            Assert.Equal(manifestPackage.GetProperty("rid").GetString(), entry.GetProperty("runtimeIdentifier").GetString());
            Assert.Equal(manifestPackage.GetProperty("platform").GetString(), entry.GetProperty("platform").GetString());
            Assert.Equal(manifestPackage.GetProperty("tensorRtLine").GetString(), entry.GetProperty("tensorRtLine").GetString());
            Assert.Equal(manifestPackage.GetProperty("tensorRtVersion").GetString(), entry.GetProperty("tensorRtVersion").GetString());
            Assert.Equal(manifestPackage.GetProperty("cudaVersion").GetString(), entry.GetProperty("cudaVersion").GetString());
            Assert.Equal(manifestPackage.GetProperty("cudnnVersion").GetString(), entry.GetProperty("cudnnVersion").GetString());
            Assert.Equal(manifestPackage.GetProperty("distributionTier").GetString(), entry.GetProperty("distributionTier").GetString());
            Assert.Equal(manifestPackage.GetProperty("validationState").GetString(), entry.GetProperty("manifestValidationState").GetString());
            Assert.Equal("clean-consumer-package-source-required", entry.GetProperty("restoreSourceMode").GetString());
            Assert.False(entry.GetProperty("usesProjectReference").GetBoolean());
            Assert.Equal(expectedNativeAssetCount, entry.GetProperty("nativeAssetCopyExpected").GetInt32());
            Assert.Equal(JsonValueKind.Null, entry.GetProperty("nativeAssetCopyActual").ValueKind);
            Assert.True(entry.GetProperty("runtimeSmokeRequired").GetBoolean());
            Assert.False(entry.GetProperty("canPromotePackageConsumerRuntimeProof").GetBoolean());
            Assert.True(entry.GetProperty("ownerActionRequired").GetBoolean());
            Assert.Contains(runtimeKey, entry.GetProperty("validatorCommand").GetString());
            Assert.Contains("-RequireExistingLog -FailOnNotProof", entry.GetProperty("validatorCommand").GetString());
            Assert.Contains("owner-action-required", entry.GetProperty("blockedReason").GetString());

            AssertArrayContainsAll(entry.GetProperty("readonlySummaryMarkersAllowed"), RequiredSummaryMarkers);
            AssertArrayContainsAll(entry.GetProperty("forbiddenProofSubstitutes"), RequiredForbiddenSubstitutes);
        }
    }

    [Fact]
    public void PackageConsumerScriptRecordsRuntimeProofPreflightBoundaryFields()
    {
        string script = ReadSource("eng", "Test-PackageConsumer.ps1");

        Assert.Contains("ReadonlySummaryEvidenceKind", script);
        Assert.Contains("WrapperSurfaceEvidenceKind", script);
        Assert.Contains("IsPackageConsumerRuntimeProof", script);
        Assert.Contains("ForbiddenProofSubstitutes", script);
        Assert.Contains("RequiredRuntimeProofFields", script);
        Assert.Contains("RuntimeProofPreflight", script);
        Assert.Contains("Get-PackageConsumerRuntimeProofPreflightMatrix", script);
        Assert.Contains("Find-PackageConsumerRuntimeProofPreflightEntry", script);
        Assert.Contains("package-consumer-runtime-proof-preflight-matrix.json", script);
        Assert.Contains("Runtime Proof Preflight Boundary", script);
        Assert.Contains("preflight-entry-missing-or-matrix-not-found", script);
        Assert.Contains("readonly-summary-diagnostics-not-runtime-proof", script);
        Assert.Contains("package-consumer-wrapper-surface-diagnostics", script);
        Assert.Contains("$isPackageConsumerRuntimeProof = $false", script);
        Assert.Contains("SmokeResult=passed without strict validator", script);
        Assert.Contains("`IsPackageConsumerRuntimeProof=True` is reserved", script);
        Assert.Contains("EngineDeploymentSummary=", script);
        Assert.Contains("GraphDiagnosticSummary=", script);
        Assert.Contains("MemoryRangeSummary=", script);
    }

    [Fact]
    public void ExternalRuntimeProofValidatorRequiresPreflightAlignmentBeforePromotion()
    {
        string script = ReadSource("eng", "Test-ExternalRuntimeProofRecord.ps1");

        Assert.Contains("Get-PackageConsumerRuntimeProofPreflightMatrix", script);
        Assert.Contains("Find-PackageConsumerRuntimeProofPreflightEntry", script);
        Assert.Contains("package-consumer-runtime-proof-preflight-matrix.json", script);
        Assert.Contains("runtimeProofPreflight", script);
        Assert.Contains("preflight-matrix-found", script);
        Assert.Contains("preflight-runtime-entry", script);
        Assert.Contains("preflight-runtime-package-id", script);
        Assert.Contains("preflight-restore-source-mode", script);
        Assert.Contains("preflight-native-assets-expected", script);
        Assert.Contains("preflight-native-assets-found", script);
        Assert.Contains("preflight-boundary-not-promotable", script);
        Assert.Contains("packageSource.runtimePackageId", script);
        Assert.Contains("packageSource.restoreSourceMode", script);
        Assert.Contains("results.nativeAssetsExpected", script);
        Assert.Contains("results.nativeAssetsFound", script);
        Assert.Contains("RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source.", script);
    }

    [Fact]
    public void ExternalRuntimeProofTemplatesCarryPreflightAlignmentFields()
    {
        string template = ReadSource("eng", "Export-ExternalRuntimeProofRecordTemplate.ps1");
        string inputTemplate = ReadSource("eng", "Export-ExternalRuntimeProofRecordInputTemplate.ps1");
        string example = ReadSource("eng", "Export-ExternalRuntimeProofRecordExample.ps1");
        string combined = template + inputTemplate + example;

        Assert.Contains("package-consumer-runtime-proof-preflight-matrix.json", combined);
        Assert.Contains("Find-PackageConsumerRuntimeProofPreflightEntry", combined);
        Assert.Contains("runtimeProofPreflight", combined);
        Assert.Contains("packageSource.runtimePackageId", combined);
        Assert.Contains("packageSource.restoreSourceMode", combined);
        Assert.Contains("results.nativeAssetsExpected", combined);
        Assert.Contains("results.nativeAssetsFound", combined);
        Assert.Contains("RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source.", combined);
        Assert.Contains("canPromotePackageConsumerRuntimeProof = $false", combined);
    }

    [Fact]
    public void ReleasePackageProofBundleRequiresExternalPreflightAlignmentWithoutPromotingPreflight()
    {
        string script = ReadSource("eng", "Export-ReleasePackageProofBundle.ps1");

        Assert.Contains("externalRuntimeProofPreflightAligned", script);
        Assert.Contains("externalRuntimeProofPreflightMatrixFound", script);
        Assert.Contains("externalRuntimeProofPreflightEntryFound", script);
        Assert.Contains("externalRuntimeProofPreflightRuntimePackageIdMatches", script);
        Assert.Contains("externalRuntimeProofPreflightRestoreSourceModeMatches", script);
        Assert.Contains("externalRuntimeProofPreflightNativeAssetsExpectedMatches", script);
        Assert.Contains("externalRuntimeProofPreflightNativeAssetsFoundMatches", script);
        Assert.Contains("package-consumer-runtime-proof-preflight-matrix.json", script);
        Assert.Contains("preflightAligned=", script);
        Assert.Contains("RuntimeProofPreflight itself is not proof", script);
        Assert.Contains("RuntimeProofPreflight alignment is a strict validator prerequisite", script);
        Assert.Contains("$canPromoteRuntimeProof = $false", script);
    }

    [Fact]
    public void DocumentationKeepsPreflightAndSummaryMarkersOutOfRuntimeProof()
    {
        string matrixDoc = ReadSource("docs", "articles", "zh-cn", "package-consumer-runtime-proof-preflight-matrix.md");
        string validationDoc = ReadSource("docs", "articles", "zh-cn", "package-consumer-validation.md");
        string playbook = ReadSource("docs", "articles", "zh-cn", "package-consumer-runtime-proof-playbook.md");

        Assert.Contains("不是 runtime proof", matrixDoc);
        Assert.Contains("clean consumer runtime proof", matrixDoc);
        Assert.Contains("strict validator", matrixDoc);
        Assert.Contains("blocked-by-cuda-driver", matrixDoc);
        Assert.Contains("bridge-only", matrixDoc);
        Assert.Contains("local feed", matrixDoc);
        Assert.Contains("ReadonlySummaryEvidenceKind", validationDoc);
        Assert.Contains("IsPackageConsumerRuntimeProof", validationDoc);
        Assert.Contains("Package Consumer Runtime Proof 预检矩阵", validationDoc);
        Assert.Contains("readonly summary", playbook);
        Assert.Contains("bridge-only", playbook);
        Assert.Contains("strict validator", playbook);
    }

    private static JsonElement FindEntry(JsonElement entries, string runtimeKey)
    {
        foreach (JsonElement entry in entries.EnumerateArray())
        {
            if (entry.GetProperty("runtimePackageKey").GetString() == runtimeKey)
            {
                return entry;
            }
        }

        throw new InvalidOperationException("Runtime key not found: " + runtimeKey);
    }

    private static void AssertArrayContainsAll(JsonElement array, IEnumerable<string> expectedValues)
    {
        string[] actual = array.EnumerateArray()
            .Select(item => item.GetString() ?? string.Empty)
            .ToArray();

        foreach (string expected in expectedValues)
        {
            Assert.Contains(expected, actual);
        }
    }

    private static JsonElement ReadMatrix()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(MatrixPath()));
        return document.RootElement.Clone();
    }

    private static JsonElement[] ReadRuntimeManifestPackages()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.GetProperty("packages")
            .EnumerateArray()
            .Select(static package => package.Clone())
            .ToArray();
    }

    private static JsonElement FindManifestPackage(IEnumerable<JsonElement> packages, string runtimeKey)
    {
        foreach (JsonElement package in packages)
        {
            if (package.GetProperty("key").GetString() == runtimeKey)
            {
                return package;
            }
        }

        throw new InvalidOperationException("Runtime manifest package not found: " + runtimeKey);
    }

    private static int CountExpectedNativeAssets(JsonElement package)
    {
        return 1 +
            package.GetProperty("tensorRtFiles").GetArrayLength() +
            package.GetProperty("cudaFiles").GetArrayLength() +
            package.GetProperty("cudnnFiles").GetArrayLength();
    }

    private static string MatrixPath()
    {
        return Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-runtime-proof-preflight-matrix.json");
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
