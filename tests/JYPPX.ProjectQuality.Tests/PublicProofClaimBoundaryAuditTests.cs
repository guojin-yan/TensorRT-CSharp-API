using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicProofClaimBoundaryAuditTests
{
    [Fact]
    public void PublicClaimAuditPassesWithoutPromotingProof()
    {
        RunPowerShell("Test-PublicProofClaimBoundaryAudit.ps1", "-Strict");

        using JsonDocument document = ReadFinalReleaseJson("public-proof-claim-boundary-audit.json");
        JsonElement audit = document.RootElement;

        Assert.Equal("public-proof-claim-boundary-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("public-proof-claim-boundary-audit-passed", audit.GetProperty("auditState").GetString());
        Assert.Equal("public-docs-proof-boundary-freeze-passed", audit.GetProperty("publicFreezeState").GetString());
        Assert.True(audit.GetProperty("scannedFileCount").GetInt32() > 0);
        Assert.True(audit.GetProperty("publicFreezeRequiredCount").GetInt32() >= 5);
        Assert.Equal(0, audit.GetProperty("publicFreezeFindingCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("blockedFindingCount").GetInt32());
        Assert.Equal("inline-plus-markdown-heading-stack", audit.GetProperty("negationContextMode").GetString());
        AssertFlagsStayNonProof(audit);
        AssertBoundary(audit.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void MarkdownNegativeSectionsAreSafeButRealPromotionStillFailsClosed()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), $"public-proof-boundary-{Guid.NewGuid():N}");
        string docsRoot = Path.Combine(fixtureRoot, "docs");
        string outputRoot = Path.Combine(fixtureRoot, "output");
        Directory.CreateDirectory(docsRoot);
        Directory.CreateDirectory(outputRoot);

        try
        {
            File.WriteAllText(Path.Combine(docsRoot, "claims.md"), """
                # Claims

                ## Cannot claim

                - local feed is published.

                ## 以下材料不得替代发布证明

                - failedBlockerCount=0 means ready to publish.

                ## Current release status

                - candidate 可发布性 audit is active.
                - local feed is published.
                - failedBlockerCount=0 means ready to publish.
                """);

            RunPowerShell(
                "Test-PublicProofClaimBoundaryAudit.ps1",
                "-RepositoryRoot",
                fixtureRoot,
                "-OutputDirectory",
                outputRoot);

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(
                Path.Combine(outputRoot, "public-proof-claim-boundary-audit.json")));
            JsonElement audit = document.RootElement;
            JsonElement[] findings = audit.GetProperty("findings").EnumerateArray().ToArray();

            Assert.Equal("inline-plus-markdown-heading-stack", audit.GetProperty("negationContextMode").GetString());
            Assert.Contains(findings, static item =>
                item.GetProperty("id").GetString() == "non-proof-artifact-promoted" &&
                item.GetProperty("line").GetInt32() == 14);
            Assert.Contains(findings, static item =>
                item.GetProperty("id").GetString() == "failed-blocker-zero-promoted" &&
                item.GetProperty("line").GetInt32() == 15);
            Assert.DoesNotContain(findings, static item =>
                item.GetProperty("path").GetString() == "docs/claims.md" &&
                item.GetProperty("line").GetInt32() is 5 or 9 or 13);
        }
        finally
        {
            Directory.Delete(fixtureRoot, recursive: true);
        }
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName);
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static void AssertFlagsStayNonProof(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start PowerShell.");
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(
            process.ExitCode == 0,
            $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
    }
}
