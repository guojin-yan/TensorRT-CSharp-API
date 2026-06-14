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
        string samplesRoot = Path.Combine(RepositoryPaths.Root, "samples");
        string[] sampleDirectories = Directory.GetDirectories(samplesRoot)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .Where(name => !string.Equals(name, "JYPPX.SampleSupport", StringComparison.Ordinal))
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.DoesNotContain(sampleDirectories, name => name.Contains("SmokeRunner", StringComparison.OrdinalIgnoreCase));

        foreach (string sampleDirectory in sampleDirectories)
        {
            string directory = Path.Combine(samplesRoot, sampleDirectory);
            string[] projects = Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly);

            Assert.Single(projects);
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
}
