using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DocumentationPortabilityTests
{
    private static readonly Regex MachineSpecificPath = new(
        @"(?:[DE]:\\|C:\\Users\\guoji)",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    [Fact]
    public void ChineseArticlesDoNotEmbedMachineSpecificDrivePaths()
    {
        string articleRoot = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn");
        foreach (string path in Directory.EnumerateFiles(articleRoot, "*.md", SearchOption.AllDirectories))
        {
            string content = File.ReadAllText(path);
            Assert.False(
                MachineSpecificPath.IsMatch(content),
                $"Chinese article contains a machine-specific drive path: {Path.GetRelativePath(RepositoryPaths.Root, path)}");
        }
    }
}
