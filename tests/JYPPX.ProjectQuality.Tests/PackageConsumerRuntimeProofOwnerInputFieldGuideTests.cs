using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PackageConsumerRuntimeProofOwnerInputFieldGuideTests
{
    [Fact]
    public void OwnerInputFieldGuideDocumentsRequiredRuntimeProofFieldsAndForbiddenSubstitutes()
    {
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-runtime-proof-owner-input-field-guide.md");
        string article = File.ReadAllText(articlePath);
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-owner-input-field-guide.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-owner-input-field-guide.md", docsToc, StringComparison.Ordinal);

        foreach (string field in new[]
        {
            "ownerName",
            "machineName",
            "gpuName",
            "cudaDriverSupportedRuntime",
            "cudnnVersion",
            "tensorRtLine",
            "restoreCommand",
            "buildCommand",
            "exitCode",
            "startedAtUtc",
            "finishedAtUtc",
            "dependencyProbeStatus",
            "smokeStatus",
            "nativeAssetsCopied",
            "smokeLogSha256",
            "publicPackageSourceKind",
            "publicPackageFeedUrl",
            "managedPackageUrl",
            "runtimePackageUrl",
            "sourceRunnerQueueStatus",
            "sourceRunnerInfrastructureStatus",
            "sourceRunnerOwnerAction",
            "stdoutSummary",
            "stderrSummary",
            "failureDiagnostic"
        })
        {
            Assert.Contains(field, article, StringComparison.Ordinal);
        }

        foreach (string forbidden in new[]
        {
            "template-only",
            "draft JSON",
            "local `.nupkg`",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "queued GitHub Actions run",
            "missing self-hosted runner",
            "build-only",
            "parse-only",
            "dry-run",
            "dependency-probe-only",
            "GUI 截图",
            "TensorRtExec build report"
        })
        {
            Assert.Contains(forbidden, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict", article, StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofOwnerInputSchema.ps1", article, StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1", article, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-owner-input.schema.json", article, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-forbidden-substitute-scan.json", article, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
        Assert.Contains("本阶段执行清单", article, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SmokeRuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RunSmoke -KeepConsumerOutput", article, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts\\final-release\\external-runtime-proof-record.json", article, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog -FailOnNotProof", article, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", article, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", article, StringComparison.Ordinal);
        Assert.Contains("no-stderr-emitted", article, StringComparison.Ordinal);
        Assert.Contains("validationState=real-runtime-proof", article, StringComparison.Ordinal);
    }
}
