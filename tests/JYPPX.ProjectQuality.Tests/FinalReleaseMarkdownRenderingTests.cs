using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalReleaseMarkdownRenderingTests
{
    private static readonly string[] KeyOwnerHandoffMarkdownFiles =
    {
        "external-clean-consumer-execution-result-import.md",
        "post-publish-clean-consumer-proof-result-import.md",
        "final-owner-execution-checklist.md",
        "final-owner-execution-one-screen-pack.md",
        "final-owner-proof-action-worklist.md",
        "final-owner-execution-package.md",
        "final-post-publish-clean-consumer-proof-record-contract.md",
        "final-release-close-owner-approval-contract.md",
        "owner-public-publish-execution-result-input-contract.md",
        "owner-real-input-landing-pack.md",
        "owner-real-evidence-end-to-end-release-gate.md",
        "final-public-publish-acceptance-gate.md",
        "release-evidence-bundle.md",
    };

    [Fact]
    public void GeneratedReleaseMarkdownFilesDoNotRenderNestedArraysAsSystemObject()
    {
        string[] markdownFiles =
        [
            .. Directory.GetFiles(
                Path.Combine(RepositoryPaths.Root, "artifacts", "final-release"),
                "*.md",
                SearchOption.TopDirectoryOnly),
            .. Directory.GetFiles(
                Path.Combine(RepositoryPaths.Root, "artifacts", "package-consumer"),
                "*.md",
                SearchOption.TopDirectoryOnly),
            .. Directory.GetFiles(
                Path.Combine(RepositoryPaths.Root, "artifacts", "linux-dry-run"),
                "*.md",
                SearchOption.AllDirectories),
        ];

        Assert.NotEmpty(markdownFiles);

        string[] failures = markdownFiles
            .Where(static path => File.ReadAllText(path).Contains("System.Object[]", StringComparison.Ordinal))
            .Select(static path => Path.GetRelativePath(RepositoryPaths.Root, path))
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.True(failures.Length == 0, "Nested arrays leaked into release Markdown: " + string.Join(", ", failures));
    }

    [Fact]
    public void ReleaseFreezeCarriesTheGeneratedPackageProofState()
    {
        string finalReleaseDirectory = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release");
        using JsonDocument packageProof = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(finalReleaseDirectory, "release-package-proof-bundle.json")));
        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(finalReleaseDirectory, "release-freeze-final-verification.json")));

        string expected = packageProof.RootElement.GetProperty("proofState").GetString()!;
        string actual = freeze.RootElement.GetProperty("releasePackageProofBundleState").GetString()!;

        Assert.Equal(expected, actual);
        Assert.NotEqual("missing-release-package-proof-bundle", actual);
    }

    [Fact]
    public void OwnerHandoffMarkdownKeepsBlockedNonProofAndOwnerActionBoundaries()
    {
        string finalReleaseDirectory = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release");
        string combined = string.Empty;

        foreach (string fileName in KeyOwnerHandoffMarkdownFiles)
        {
            string path = Path.Combine(finalReleaseDirectory, fileName);
            Assert.True(File.Exists(path), $"Expected owner handoff Markdown to exist: {fileName}");

            string markdown = File.ReadAllText(path);
            Assert.DoesNotContain("System.Object[]", markdown, StringComparison.Ordinal);
            Assert.Contains("blocked", markdown, StringComparison.OrdinalIgnoreCase);

            combined += markdown;
        }

        Assert.True(
            combined.Contains("not proof", StringComparison.OrdinalIgnoreCase) ||
            combined.Contains("non-proof", StringComparison.OrdinalIgnoreCase) ||
            combined.Contains("failedBlockerCountIsNotProof", StringComparison.OrdinalIgnoreCase),
            "Expected non-proof boundary markers in final-release handoff Markdown.");
        Assert.True(
            combined.Contains("owner-action-required", StringComparison.OrdinalIgnoreCase) ||
            combined.Contains("Owner Action Required", StringComparison.OrdinalIgnoreCase),
            "Expected owner action boundary markers in final-release handoff Markdown.");
        Assert.Contains("canPublishPublicly: `False`", combined, StringComparison.OrdinalIgnoreCase);
    }
}
