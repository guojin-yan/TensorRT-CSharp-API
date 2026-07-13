using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionFreezeTests
{
    [Fact]
    public void OwnerExecutionFreezeAndFinalCloseDecisionRemainBlockedAndAuditable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerReleaseExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueFinalCloseDecisionTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueFinalCloseDecision.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalEvidenceFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalEvidenceFreeze.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument ownerValidationDocument = ReadFinalReleaseJson("owner-release-execution-package-validation.json");
        JsonElement ownerValidation = ownerValidationDocument.RootElement;
        Assert.Equal("owner-release-execution-package-validation", ownerValidation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-execution-package-ready", ownerValidation.GetProperty("validationState").GetString());
        Assert.Equal("blocked-real-proof-required", ownerValidation.GetProperty("packageState").GetString());
        Assert.Equal(0, ownerValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(ownerValidation.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerValidation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerValidation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument closeDecisionDocument = ReadFinalReleaseJson("release-issue-final-close-decision-validation.json");
        JsonElement closeDecision = closeDecisionDocument.RootElement;
        Assert.Equal("release-issue-final-close-decision-validation", closeDecision.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-final-close-decision-required", closeDecision.GetProperty("validationState").GetString());
        Assert.Equal(0, closeDecision.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(closeDecision.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(closeDecision.GetProperty("performsPublish").GetBoolean());
        Assert.False(closeDecision.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(closeDecision.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument freezeDocument = ReadFinalReleaseJson("final-evidence-freeze.json");
        JsonElement freeze = freezeDocument.RootElement;
        Assert.Equal("final-evidence-freeze", freeze.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-evidence-frozen-owner-action-required", freeze.GetProperty("freezeState").GetString());
        Assert.Equal(0, freeze.GetProperty("missingSourceArtifactCount").GetInt32());
        Assert.Equal(0, freeze.GetProperty("missingHashCount").GetInt32());
        Assert.Equal(5, freeze.GetProperty("packageEvidenceClassificationContractCount").GetInt32());
        Assert.Equal(3, freeze.GetProperty("rejectedPackageEvidenceKindCount").GetInt32());
        Assert.Equal(2, freeze.GetProperty("acceptedPublicPackageEvidenceKindCount").GetInt32());
        AssertPackageEvidenceClass(freeze, "local-feed-proof", acceptedAsPublicProof: false);
        AssertPackageEvidenceClass(freeze, "direct-nupkg-proof", acceptedAsPublicProof: false);
        AssertPackageEvidenceClass(freeze, "project-reference-proof", acceptedAsPublicProof: false);
        AssertPackageEvidenceClass(freeze, "public-package-download-proof", acceptedAsPublicProof: true);
        AssertPackageEvidenceClass(freeze, "post-publish-clean-consumer-proof", acceptedAsPublicProof: true);
        Assert.False(freeze.GetProperty("performsPublish").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument freezeValidationDocument = ReadFinalReleaseJson("final-evidence-freeze-validation.json");
        JsonElement freezeValidation = freezeValidationDocument.RootElement;
        Assert.Equal("final-evidence-freeze-validation", freezeValidation.GetProperty("recordKind").GetString());
        Assert.Equal("final-evidence-freeze-valid-blocked-owner-action-required", freezeValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, freezeValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(5, freezeValidation.GetProperty("packageEvidenceClassificationContractCount").GetInt32());
        Assert.Equal(3, freezeValidation.GetProperty("rejectedPackageEvidenceKindCount").GetInt32());
        Assert.Equal(2, freezeValidation.GetProperty("acceptedPublicPackageEvidenceKindCount").GetInt32());
        AssertValidationItemPassed(freezeValidation, "package-evidence-classification-contract");
        AssertValidationItemPassed(freezeValidation, "rejected-package-substitutes");
        AssertValidationItemPassed(freezeValidation, "accepted-public-package-proof-kinds");
        Assert.False(freezeValidation.GetProperty("performsPublish").GetBoolean());
        Assert.False(freezeValidation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freezeValidation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Equal("owner-execution-package-ready", bundle.GetProperty("ownerReleaseExecutionPackageValidationState").GetString());
        Assert.Equal("blocked-evidence-frozen-owner-action-required", bundle.GetProperty("finalEvidenceFreezeState").GetString());
        Assert.Equal("final-evidence-freeze-valid-blocked-owner-action-required", bundle.GetProperty("finalEvidenceFreezeValidationState").GetString());
        Assert.Equal(5, bundle.GetProperty("finalEvidenceFreezePackageEvidenceClassificationContractCount").GetInt32());
        Assert.Equal(3, bundle.GetProperty("finalEvidenceFreezeRejectedPackageEvidenceKindCount").GetInt32());
        Assert.Equal(2, bundle.GetProperty("finalEvidenceFreezeAcceptedPublicPackageEvidenceKindCount").GetInt32());
        Assert.Equal("blocked-owner-final-close-decision-required", bundle.GetProperty("releaseIssueFinalCloseDecisionValidationState").GetString());
        Assert.False(bundle.GetProperty("performsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] evidenceItemIds = bundle.GetProperty("evidenceItems").EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("owner-release-execution-package-validation", evidenceItemIds);
        Assert.Contains("final-evidence-freeze", evidenceItemIds);
        Assert.Contains("release-issue-final-close-decision", evidenceItemIds);
        JsonElement freezeEvidenceItem = bundle.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "final-evidence-freeze");
        string freezeEvidenceState = freezeEvidenceItem.GetProperty("state").GetString()!;
        Assert.Contains("packageEvidenceClassifications=5", freezeEvidenceState, StringComparison.Ordinal);
        Assert.Contains("rejectedPackageEvidenceKinds=3", freezeEvidenceState, StringComparison.Ordinal);
        Assert.Contains("acceptedPublicPackageEvidenceKinds=2", freezeEvidenceState, StringComparison.Ordinal);

        string[] sourceArtifacts = bundle.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-release-execution-package-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-evidence-freeze.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-evidence-freeze-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-final-close-decision-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-release-execution-package-validation.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-evidence-freeze.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-final-close-decision.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-release-execution-package-validation.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-evidence-freeze.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-final-close-decision.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-release-execution-package-validation.json", readme, StringComparison.Ordinal);
        Assert.Contains("final-evidence-freeze.json", readmeZh, StringComparison.Ordinal);
        Assert.Contains("release issue final close decision validation: `blocked-owner-final-close-decision-required`", evidenceMarkdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertPackageEvidenceClass(JsonElement freeze, string id, bool acceptedAsPublicProof)
    {
        JsonElement evidenceClass = freeze.GetProperty("packageEvidenceClassificationContract")
            .EnumerateArray()
            .Single(candidate => candidate.GetProperty("id").GetString() == id);
        Assert.Equal(acceptedAsPublicProof, evidenceClass.GetProperty("acceptedAsPublicProof").GetBoolean());
        Assert.False(string.IsNullOrWhiteSpace(evidenceClass.GetProperty("requiredValidator").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(evidenceClass.GetProperty("reason").GetString()));
    }

    private static void AssertValidationItemPassed(JsonElement validation, string id)
    {
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean());
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
