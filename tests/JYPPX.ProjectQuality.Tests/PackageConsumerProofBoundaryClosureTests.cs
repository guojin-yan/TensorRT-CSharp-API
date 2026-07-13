using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerProofBoundaryClosureTests
{
    [Fact]
    public void PackageConsumerProofTemplatesCandidatesAndBundleKeepNonProofBoundary()
    {
        string ownerInputTemplateExporter = ReadText("eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1");
        string ownerInputValidator = ReadText("eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1");
        string recordTemplateExporter = ReadText("eng", "Export-PackageConsumerRuntimeProofRecordTemplate.ps1");
        string recordValidator = ReadText("eng", "Test-PackageConsumerRuntimeProofRecord.ps1");
        string candidateExporter = ReadText("eng", "Export-PackageConsumerRuntimeProofCandidate.ps1");
        string candidateValidator = ReadText("eng", "Test-PackageConsumerRuntimeProofCandidate.ps1");
        string evidenceExporter = ReadText("eng", "Export-ReleaseEvidenceBundle.ps1");
        string combined = ownerInputTemplateExporter + ownerInputValidator + recordTemplateExporter + recordValidator + candidateExporter + candidateValidator + evidenceExporter;

        foreach (string marker in new[]
        {
            "package-consumer-runtime-proof-owner-input",
            "template-owner-input-required",
            "blocked-owner-input-required",
            "package-consumer-runtime-proof-record-template",
            "package-consumer-runtime-proof-record-validation",
            "template-only",
            "package-consumer-runtime-proof-candidate",
            "blocked-real-package-consumer-smoke-required",
            "packageConsumerRuntimeProofRecordCanPromoteRuntimeProof",
            "packageConsumerRuntimeProofOwnerInputValidationState",
            "packageConsumerRuntimeProofRecordValidationState",
            "--runtime-package-key"
        })
        {
            Assert.Contains(marker, combined, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "canPromoteRuntimeProof = $false",
            "canPromoteProof = $false",
            "performsPublish = $false",
            "canPublishPublicly = $false",
            "canCloseReleaseIssue = $false"
        })
        {
            Assert.Contains(marker, combined, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void PackageConsumerProofScriptsRejectLocalSubstitutesAndPublishClaims()
    {
        string ownerInputValidator = ReadText("eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1");
        string recordValidator = ReadText("eng", "Test-PackageConsumerRuntimeProofRecord.ps1");
        string candidateValidator = ReadText("eng", "Test-PackageConsumerRuntimeProofCandidate.ps1");
        string ownerInputTemplateExporter = ReadText("eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1");
        string recordTemplateExporter = ReadText("eng", "Export-PackageConsumerRuntimeProofRecordTemplate.ps1");
        string candidateExporter = ReadText("eng", "Export-PackageConsumerRuntimeProofCandidate.ps1");
        string combined = ownerInputValidator + recordValidator + candidateValidator + ownerInputTemplateExporter + recordTemplateExporter + candidateExporter;

        foreach (string marker in new[]
        {
            "cleanExternalConsumerRoot",
            "publicPackageSource",
            "no-project-reference-to-repository",
            "public-package-source-not-local",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "canPublishPublicly",
            "canCloseReleaseIssue",
            "canPromoteRuntimeProof",
            "canPromoteProof",
            "performsPublish"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", combined, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("canPublishPublicly = $false", combined, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", combined, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", combined, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageConsumerProofDocsStateOwnerInputAndNonSubstituteBoundaries()
    {
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");
        string recordArticle = ReadText("docs", "articles", "zh-cn", "package-consumer-runtime-proof-record.md");
        string ownerInputArticle = ReadText("docs", "articles", "zh-cn", "package-consumer-runtime-proof-owner-input.md");
        string worklistArticle = ReadText("docs", "articles", "zh-cn", "package-consumer-runtime-proof-worklist.md");
        string validationArticle = ReadText("docs", "articles", "zh-cn", "package-consumer-validation.md");
        string combined = recordArticle + ownerInputArticle + worklistArticle + validationArticle;

        foreach (string article in new[]
        {
            "articles/zh-cn/package-consumer-runtime-proof-record.md",
            "articles/zh-cn/package-consumer-runtime-proof-owner-input.md",
            "articles/zh-cn/package-consumer-runtime-proof-worklist.md",
            "articles/zh-cn/package-consumer-validation.md"
        })
        {
            Assert.Contains(article, docsIndex, StringComparison.Ordinal);
            Assert.Contains(article, docsToc, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "package-consumer-runtime",
            "clean external consumer",
            "local feed",
            "ProjectReference",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.True(
            combined.Contains("not proof", StringComparison.OrdinalIgnoreCase) ||
            combined.Contains("不是 proof", StringComparison.OrdinalIgnoreCase),
            "Package consumer proof docs must state that templates, candidates, local feed, ProjectReference, or direct nupkg are not proof.");

        Assert.DoesNotContain("dotnet nuget push", combined, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
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
