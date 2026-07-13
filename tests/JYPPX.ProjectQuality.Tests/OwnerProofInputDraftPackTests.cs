using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerProofInputDraftPackTests
{
    private static readonly string[] RequiredLineIds =
    {
        "owner-authorization",
        "package-consumer-runtime",
        "linux-runner-proof",
        "real-model-runtime",
        "post-publish-verification",
        "release-issue-close-record",
    };

    [Fact]
    public void OwnerProofInputDraftPackExportsStrictNonProofDraftSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofInputPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputDraftPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofInputDraftPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument draftDocument = ReadFinalReleaseJson("owner-proof-input-draft-pack.json");
        JsonElement root = draftDocument.RootElement;

        Assert.Equal("owner-proof-input-draft-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-draft-non-proof", root.GetProperty("draftPackState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("draftSpecCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("blockedDraftSpecCount").GetInt32());
        Assert.True(root.GetProperty("placeholderFieldCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("sha256FieldCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("existingFileFieldCount").GetInt32() >= 1);

        string[] ids = root.GetProperty("draftSpecs")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(RequiredLineIds.Order(StringComparer.Ordinal).ToArray(), ids);

        foreach (JsonElement spec in root.GetProperty("draftSpecs").EnumerateArray())
        {
            Assert.Equal("blocked-draft-non-proof", spec.GetProperty("draftState").GetString());
            Assert.False(spec.GetProperty("inputDraftIsProof").GetBoolean());
            Assert.False(spec.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(spec.GetProperty("performsPublish").GetBoolean());
            Assert.False(spec.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(spec.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(spec.GetProperty("inputDraftPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(spec.GetProperty("strictValidationCommand").GetString()));

            string[] markers = spec.GetProperty("blockedByNonProofMarkers")
                .EnumerateArray()
                .Select(static marker => marker.GetString()!)
                .ToArray();
            Assert.Contains("template", markers);
            Assert.Contains("draft", markers);
            Assert.Contains("ProjectReference", markers);
            Assert.Contains("local feed", markers);
            Assert.Contains("direct .nupkg reference", markers);
            Assert.Contains("release-issue-close-record-template.json", markers);
        }

        JsonElement packageConsumer = root.GetProperty("draftSpecs")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        string[] packageRules = packageConsumer.GetProperty("requiredRealInputRules")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string rule in new[]
        {
            "cleanExternalConsumerIdentity",
            "noProjectReference",
            "noLocalFeedAsPublicProof",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "runtimePackageKeyMatches",
            "compatibleHostMetadata",
            "smokeCommandIncludesRuntimePackageKey",
            "smokeLogPath",
            "smokeLogSha256",
        })
        {
            Assert.Contains(rule, packageRules);
        }
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", packageConsumer.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", packageConsumer.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);

        JsonElement closeRecord = root.GetProperty("draftSpecs")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        string[] closeRules = closeRecord.GetProperty("requiredRealInputRules")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string rule in new[]
        {
            "releaseEvidenceBundleSha256",
            "releaseClosePreflightPathAndHash",
            "staleClaimsAuditPathAndHash",
            "postPublishProofValidationPathAndHash",
            "rollbackPlan",
            "ownerFinalCloseDecision",
            "strictCloseValidatorCommand",
        })
        {
            Assert.Contains(rule, closeRules);
        }
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", closeRecord.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotCloseReady", closeRecord.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-proof-input-draft-pack-validation.json");
        JsonElement validationRoot = validationDocument.RootElement;
        Assert.Equal("owner-proof-input-draft-pack-validation", validationRoot.GetProperty("recordKind").GetString());
        Assert.Equal("valid-draft-non-proof", validationRoot.GetProperty("validationState").GetString());
        Assert.True(validationRoot.GetProperty("isValidDraftPack").GetBoolean());
        Assert.False(validationRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(0, validationRoot.GetProperty("promotedProofItemCount").GetInt32());

        using JsonDocument evidenceBundle = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidenceBundle.RootElement;
        Assert.Equal("blocked-draft-non-proof", evidenceRoot.GetProperty("ownerProofInputDraftPackState").GetString());
        Assert.Equal("valid-draft-non-proof", evidenceRoot.GetProperty("ownerProofInputDraftPackValidationState").GetString());
        Assert.Equal(RequiredLineIds.Length, evidenceRoot.GetProperty("ownerProofInputDraftPackItemCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, evidenceRoot.GetProperty("ownerProofInputDraftPackBlockedItemCount").GetInt32());
        Assert.False(evidenceRoot.GetProperty("ownerProofInputDraftPackPerformsPublish").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("ownerProofInputDraftPackCanPublishPublicly").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("ownerProofInputDraftPackCanCloseReleaseIssue").GetBoolean());

        JsonElement draftEvidenceItem = evidenceRoot.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-proof-input-draft-pack");
        Assert.False(draftEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked-draft-non-proof", draftEvidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("non-proof drafting surface", draftEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceEvidence = evidenceRoot.GetProperty("sourceEvidence")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] sourceArtifacts = evidenceRoot.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string artifactPath in new[]
        {
            "artifacts/final-release/owner-proof-input-draft-pack.json",
            "artifacts/final-release/owner-proof-input-draft-pack.md",
            "artifacts/final-release/owner-proof-input-draft-pack-validation.json",
            "artifacts/final-release/owner-proof-input-draft-pack-validation.md",
        })
        {
            Assert.Contains(artifactPath, sourceEvidence);
            Assert.Contains(artifactPath, sourceArtifacts);
        }

        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));
        Assert.Contains("owner-proof-input-draft-pack", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner proof input draft blocked specs: `6`", evidenceMarkdown, StringComparison.Ordinal);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-input-draft-pack.md"));
        string releaseEvidenceArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-proof-input-draft-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-input-draft-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Owner proof input draft pack: `artifacts/final-release/owner-proof-input-draft-pack.md`", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Owner proof input draft pack artifact: `artifacts/final-release/owner-proof-input-draft-pack.json`", readme, StringComparison.Ordinal);
        Assert.Contains("Owner proof input draft pack artifact：`artifacts/final-release/owner-proof-input-draft-pack.json`", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner-proof-input-draft-pack", releaseEvidenceArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "recordKind=owner-proof-input-draft-pack",
            "draftPackState=blocked-draft-non-proof",
            "inputDraftIsProof=false",
            "canPromoteProof=false",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "release-issue-close-record",
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
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
