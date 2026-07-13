using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofClaimAuditExpansionTests
{
    [Fact]
    public void ReadmesKeepCleanConsumerClosurePacksAsBlockedNonProofOwnerActions()
    {
        string readme = ReadSource("README.md");
        string zhReadme = ReadSource("README.zh-CN.md");

        foreach (string text in new[] { readme, zhReadme })
        {
            Assert.Contains("clean-consumer-proof-execution-bundle", text, StringComparison.Ordinal);
            Assert.Contains("clean-consumer-external-proof-closure-pack", text, StringComparison.Ordinal);
            Assert.Contains("blocked", text, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("non-proof", text, StringComparison.OrdinalIgnoreCase);
            AssertContainsAny(text, "not run runtime smoke", "不会运行 runtime smoke");
            AssertContainsAny(text, "publish", "发布");
            AssertContainsAny(text, "close", "关闭");
            Assert.Contains("FailOnNotProof", text, StringComparison.Ordinal);
            Assert.DoesNotContain("clean-consumer-external-proof-closure-pack is runtime proof", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("clean-consumer-proof-execution-bundle is runtime proof", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("local smoke is package-consumer-runtime proof", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("local feed is post-publish proof", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("ProjectReference is package-consumer-runtime proof", text, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static void AssertContainsAny(string text, params string[] expectedValues)
    {
        Assert.True(
            expectedValues.Any(expected => text.Contains(expected, StringComparison.OrdinalIgnoreCase)),
            $"Expected one of these markers: {string.Join(", ", expectedValues)}");
    }

    private static string ReadSource(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(segments)));
    }
}
