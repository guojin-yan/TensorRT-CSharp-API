using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaDeviceInitializationLocalSmokeClassificationTests
{
    [Fact]
    public void ExportAndValidationKeepCudaDeviceInitializationSmokeAsNonProof()
    {
        RunPowerShell("Export-CudaDeviceInitializationLocalSmokeClassification.ps1");
        string output = RunPowerShell("Test-CudaDeviceInitializationLocalSmokeClassification.ps1", "-Strict");

        Assert.Contains("ValidationState=cuda-device-initialization-local-smoke-classification-validation-passed-non-proof", output, StringComparison.Ordinal);

        using JsonDocument recordDocument = ReadFinalReleaseJson("cuda-device-initialization-local-smoke-classification.json");
        JsonElement record = recordDocument.RootElement;

        Assert.Equal("cuda-device-initialization-local-smoke-classification", record.GetProperty("recordKind").GetString());
        Assert.Equal("cuda-device-initialization-local-smoke-classified-non-proof", record.GetProperty("classificationState").GetString());
        Assert.Equal("local-smoke-not-external-proof", record.GetProperty("proofKind").GetString());
        Assert.Equal("smoke/CudaDeviceInitializationProofRunner/Program.cs", record.GetProperty("sourceSmokeRunner").GetString());
        Assert.True(record.GetProperty("preInitCallOrderReady").GetBoolean());
        Assert.True(record.GetProperty("skippedTrueIsForbiddenSubstitute").GetBoolean());
        AssertFalseProofFlags(record);

        string recordText = record.GetRawText();
        Assert.Contains("Skipped=True", recordText, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof", recordText, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", record.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("cuda-device-initialization-local-smoke-classification-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("cuda-device-initialization-local-smoke-classification-validation-passed-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("findingCount").GetInt32());
        AssertFalseProofFlags(validation);
    }

    [Fact]
    public void PackageConsumerDocsAndArticleRoadmapDescribeLocalSmokeBoundary()
    {
        RunPowerShell("Export-CudaDeviceInitializationLocalSmokeClassification.ps1");
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");

        string packageConsumerDoc = ReadSource("docs", "articles", "zh-cn", "package-consumer-validation.md");
        string preflightMatrixDoc = ReadSource("docs", "articles", "zh-cn", "package-consumer-runtime-proof-preflight-matrix.md");
        string roadmapJson = ReadSource("docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.json");

        Assert.Contains("CudaDeviceInitializationProofRunner", packageConsumerDoc, StringComparison.Ordinal);
        Assert.Contains("local-smoke-not-external-proof", packageConsumerDoc, StringComparison.Ordinal);
        Assert.Contains("IsPackageConsumerRuntimeProof=False", packageConsumerDoc, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRuntimeProof=False", packageConsumerDoc, StringComparison.Ordinal);
        Assert.Contains("Skipped=True", packageConsumerDoc, StringComparison.Ordinal);

        Assert.Contains("CudaDeviceInitializationProofRunner local-smoke", preflightMatrixDoc, StringComparison.Ordinal);
        Assert.Contains("Skipped=True", preflightMatrixDoc, StringComparison.Ordinal);

        Assert.Contains("CUDA 初始化 Proof Scaffold", roadmapJson, StringComparison.Ordinal);
        Assert.Contains("CUDA Graph Event Node borrowed handle", roadmapJson, StringComparison.Ordinal);
        Assert.Contains("Package Consumer Proof 分层", roadmapJson, StringComparison.Ordinal);
        Assert.Contains("YoloVision 真实资产证据链", roadmapJson, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", roadmapJson, StringComparison.Ordinal);
        Assert.Contains("must not claim CudaDeviceInitializationProofRunner local smoke is package-consumer-runtime proof", roadmapJson, StringComparison.Ordinal);
        Assert.Contains("must not treat Skipped=True as proof", roadmapJson, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceBundleCarriesLocalSmokeClassificationAsRequiredNonProofItem()
    {
        RunPowerShell("Export-CudaDeviceInitializationLocalSmokeClassification.ps1");
        RunPowerShell("Test-CudaDeviceInitializationLocalSmokeClassification.ps1", "-Strict");
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;

        Assert.Contains(bundle.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "CudaDeviceInitializationProofRunner local smoke");
        Assert.Contains(bundle.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "local-smoke-not-external-proof");
        Assert.Contains(bundle.GetProperty("sourceArtifacts").EnumerateArray(), static artifact =>
            artifact.GetString() == "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json");

        JsonElement item = bundle.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static evidenceItem => evidenceItem.GetProperty("id").GetString() == "cuda-device-initialization-local-smoke-classification");
        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains("not runtime proof", item.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("local-smoke-not-external-proof", item.GetProperty("state").GetString()!, StringComparison.Ordinal);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        Assert.Contains(auditDocument.RootElement.GetProperty("auditedItems").EnumerateArray(), static auditedItem =>
            auditedItem.GetProperty("id").GetString() == "cuda-device-initialization-local-smoke-classification" &&
            auditedItem.GetProperty("passed").GetBoolean() == false &&
            auditedItem.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static void AssertFalseProofFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        using Process process = new();
        process.StartInfo = new ProcessStartInfo
        {
            FileName = "pwsh",
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            WorkingDirectory = RepositoryPaths.Root
        };
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"PowerShell script failed ({scriptName}) with exit code {process.ExitCode}.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }

        return stdout + stderr;
    }
}
