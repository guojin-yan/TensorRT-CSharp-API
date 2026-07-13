using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class CleanPublicPackageConsumerProofGapReportTests
{
    [Fact]
    public void GapReportCapturesOwnerActionsAndForbiddenSubstitutesWithoutPromotingProof()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanPublicPackageConsumerProofGapReport.ps1"));
        Assert.Contains("Clean public package consumer proof gap report written", output, StringComparison.Ordinal);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "clean-public-package-consumer-proof-gap-report.json")));
        JsonElement root = document.RootElement;

        Assert.Equal("clean-public-package-consumer-proof-gap-report", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-clean-public-package-consumer-proof-owner-action-required", root.GetProperty("reportState").GetString());
        Assert.Equal("blocked-owner-input-required", root.GetProperty("ownerInputValidationState").GetString());
        Assert.True(root.GetProperty("ownerInputFailedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal("template-only", root.GetProperty("recordValidationState").GetString());
        Assert.Equal(6, root.GetProperty("gapCount").GetInt32());
        Assert.Equal(6, root.GetProperty("ownerActionRequiredCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyForPromotionCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

        string[] requiredEvidence = root.GetProperty("requiredEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("publicPackageSource URI for the real package channel", requiredEvidence);
        Assert.Contains("cleanExternalConsumerRoot outside repository", requiredEvidence);
        Assert.Contains("runtime smoke command including explicit runtime package key", requiredEvidence);
        Assert.Contains("runtime smoke stdout SHA256", requiredEvidence);
        Assert.Contains("GPU name and driver version", requiredEvidence);

        string[] forbidden = root.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed", forbidden);
        Assert.Contains("ProjectReference", forbidden);
        Assert.Contains("direct nupkg", forbidden);
        Assert.Contains("build-only", forbidden);
        Assert.Contains("DependencyProbe-only", forbidden);
        Assert.Contains("bridge-only compatible-host smoke", forbidden);

        string[] gapIds = root.GetProperty("gaps").EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("missing-public-package-source", gapIds);
        Assert.Contains("missing-public-package-hashes", gapIds);
        Assert.Contains("missing-clean-external-consumer-path", gapIds);
        Assert.Contains("missing-restore-build-smoke-logs", gapIds);
        Assert.Contains("missing-log-sha256", gapIds);
        Assert.Contains("missing-host-runtime-metadata", gapIds);

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/clean-consumer-proof-execution-bundle.json", sourceArtifacts);

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "clean-public-package-consumer-proof-gap-report.md"));
        Assert.Contains("Clean Public Package Consumer Proof Gap Report", markdown, StringComparison.Ordinal);
        Assert.Contains("missing-public-package-source", markdown, StringComparison.Ordinal);
        Assert.Contains("bridge-only compatible-host smoke", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-action planning evidence only", markdown, StringComparison.OrdinalIgnoreCase);
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
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
