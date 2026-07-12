using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishVerificationOwnerInputTests
{
    [Fact]
    public void PostPublishVerificationOwnerInputExportsBlockedRecordProjectionSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationOwnerInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationOwnerInput.ps1"), "-Strict");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordFromOwnerInput.ps1"),
            "-OwnerInputPath",
            "artifacts/final-release/post-publish-verification-owner-input.template.json");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"),
            "-InputPath",
            "artifacts/final-release/post-publish-verification-record.json");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument ownerInputDocument = ReadFinalReleaseJson("post-publish-verification-owner-input.template.json");
        JsonElement ownerInput = ownerInputDocument.RootElement;
        Assert.Equal("post-publish-verification-owner-input", ownerInput.GetProperty("recordKind").GetString());
        Assert.Equal("template-owner-input-required", ownerInput.GetProperty("ownerInputState").GetString());
        Assert.False(ownerInput.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerInput.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerInput.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(ownerInput.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", ownerInput.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);
        AssertStringArrayContainsAll(
            ownerInput.GetProperty("forbiddenNonProofSubstitutes"),
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "manual approval",
            "queued GitHub Actions run",
            "missing self-hosted runner",
            "sidecar-only",
            "TensorRtExec report");

        using JsonDocument ownerValidationDocument = ReadFinalReleaseJson("post-publish-verification-owner-input-validation.json");
        JsonElement ownerValidation = ownerValidationDocument.RootElement;
        Assert.Equal("post-publish-verification-owner-input-validation", ownerValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-required", ownerValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, ownerValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(ownerValidation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(ownerValidation.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerValidation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerValidation.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(ownerValidation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("TensorRtExec report", ownerValidation.GetProperty("safetyBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument recordDocument = ReadFinalReleaseJson("post-publish-verification-record.json");
        JsonElement record = recordDocument.RootElement;
        Assert.Equal("post-publish-verification-record", record.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", record.GetProperty("verificationState").GetString());
        Assert.Equal("owner-action-required", record.GetProperty("postPublishProofClassification").GetString());
        Assert.True(record.GetProperty("ownerInputOverlayApplied").GetBoolean());
        Assert.False(record.GetProperty("performsPublish").GetBoolean());
        Assert.False(record.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(record.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(record.TryGetProperty("packagePageUrl", out _));
        Assert.True(record.TryGetProperty("downloadedManagedPackageSha256", out _));
        Assert.True(record.TryGetProperty("runtimeNativeAssetResolutionReportSha256", out _));
        Assert.True(record.TryGetProperty("rollbackReviewSha256", out _));
        Assert.True(record.TryGetProperty("forbiddenSubstituteScanSha256", out _));

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-verification-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("incomplete-post-publish-verification", validation.GetProperty("validationState").GetString());
        Assert.False(validation.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-input-required", evidence.GetProperty("postPublishVerificationOwnerInputValidationState").GetString());
        Assert.Equal("owner-action-required", evidence.GetProperty("postPublishVerificationRecordState").GetString());
        Assert.False(evidence.GetProperty("postPublishVerificationRecordIsProof").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishVerificationRecordCanCloseReleaseIssue").GetBoolean());

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/post-publish-verification-owner-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-owner-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-record.json", sourceArtifacts);

        Assert.Contains(evidence.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-owner-input" &&
            item.GetProperty("passed").GetBoolean() == false);
        Assert.Contains(evidence.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-record" &&
            item.GetProperty("passed").GetBoolean() == false);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "post-publish-verification-owner-input.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/post-publish-verification-owner-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-verification-owner-input.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-owner-input", article, StringComparison.Ordinal);
        Assert.Contains("post-publish verification owner input validation: `blocked-owner-input-required`", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("post-publish verification record projection: `owner-action-required`", evidenceMarkdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertStringArrayContainsAll(JsonElement array, params string[] expected)
    {
        HashSet<string> actual = array.EnumerateArray()
            .Select(item => item.GetString() ?? string.Empty)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (string item in expected)
        {
            Assert.Contains(item, actual);
        }
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
