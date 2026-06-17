using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class SampleLayoutTests
{
    private static readonly Regex ProgramClassPattern = new(
        @"(?m)^\s*(?:internal|public)?\s*(?:static\s+)?(?:sealed\s+)?(?:partial\s+)?class\s+Program\b",
        RegexOptions.CultureInvariant);

    [Fact]
    public void SamplesDirectoryContainsUserFacingProjectsOnly()
    {
        string[] sampleDirectories = GetSampleProjectDirectories();

        Assert.DoesNotContain(sampleDirectories, name => name.Contains("SmokeRunner", StringComparison.OrdinalIgnoreCase));

        string samplesRoot = Path.Combine(RepositoryPaths.Root, "samples");
        foreach (string sampleDirectory in sampleDirectories)
        {
            string directory = Path.Combine(samplesRoot, sampleDirectory);
            string[] projects = Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly);

            Assert.Single(projects);
        }
    }

    [Fact]
    public void SamplesReadmeListsEveryUserFacingProject()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));
        string[] sampleDirectories = GetSampleProjectDirectories();

        Assert.NotEmpty(sampleDirectories);
        foreach (string sampleDirectory in sampleDirectories)
        {
            Assert.Contains($"`{sampleDirectory}`", readme, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("SmokeRunner", readme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SmokeReadmeListsEveryValidationProject()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "smoke", "README.md"));
        string[] smokeDirectories = Directory.GetDirectories(Path.Combine(RepositoryPaths.Root, "smoke"))
            .Where(directory => Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly).Length == 1)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.NotEmpty(smokeDirectories);
        foreach (string smokeDirectory in smokeDirectories)
        {
            Assert.Contains($"`{smokeDirectory}`", readme, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RootReadmesDoNotAdvertiseRemovedSamplePlaceholders()
    {
        foreach (string fileName in new[] { "README.md", "README.zh-CN.md" })
        {
            string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, fileName));

            Assert.DoesNotContain("CustomKernelPreprocess", readme, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("`Classification`", readme, StringComparison.Ordinal);
            Assert.Contains("`YoloDet`", readme, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ProgramFilesDoNotUseTopLevelStatements()
    {
        string[] programFiles = Directory.EnumerateFiles(Path.Combine(RepositoryPaths.Root, "samples"), "Program.cs", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(Path.Combine(RepositoryPaths.Root, "smoke"), "Program.cs", SearchOption.AllDirectories))
            .OrderBy(static path => path, StringComparer.Ordinal)
            .ToArray();

        Assert.NotEmpty(programFiles);

        foreach (string programFile in programFiles)
        {
            string source = File.ReadAllText(programFile);
            string firstSignificantLine = GetFirstSignificantLine(source);

            Assert.Matches(ProgramClassPattern, source);
            Assert.True(
                firstSignificantLine.StartsWith("namespace ", StringComparison.Ordinal) ||
                firstSignificantLine.Contains(" class Program", StringComparison.Ordinal),
                $"Program file appears to use top-level statements: {programFile}");
        }
    }

    private static string GetFirstSignificantLine(string source)
    {
        foreach (string rawLine in source.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 ||
                line.StartsWith("using ", StringComparison.Ordinal) ||
                line.StartsWith("//", StringComparison.Ordinal) ||
                line.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }

            return line;
        }

        return string.Empty;
    }

    private static string[] GetSampleProjectDirectories()
    {
        return Directory.GetDirectories(Path.Combine(RepositoryPaths.Root, "samples"))
            .Where(directory => Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly).Length == 1)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .Where(name => !string.Equals(name, "JYPPX.SampleSupport", StringComparison.Ordinal))
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();
    }
}
