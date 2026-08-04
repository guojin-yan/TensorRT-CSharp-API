using System.Text.RegularExpressions;
using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedNamespaceRootPolicyTests
{
    private static readonly Regex NamespaceDeclaration = new(
        @"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.]*)\s*[;{]",
        RegexOptions.Compiled | RegexOptions.Multiline | RegexOptions.CultureInvariant);

    [Fact]
    public void SourceNamespacesUseOnlyTensorRtSharpOrCudaSharpProductRoots()
    {
        string sourceRoot = Path.Combine(RepositoryPaths.Root, "src");
        List<string> invalid = new();

        foreach (string path in Directory.EnumerateFiles(sourceRoot, "*.cs", SearchOption.AllDirectories))
        {
            string relativePath = Path.GetRelativePath(RepositoryPaths.Root, path);
            if (IsBuildOutput(relativePath))
            {
                continue;
            }

            foreach (Match match in NamespaceDeclaration.Matches(File.ReadAllText(path)))
            {
                string value = match.Groups[1].Value;
                if (!IsAllowedProductNamespace(value))
                {
                    invalid.Add($"{relativePath}: {value}");
                }
            }
        }

        Assert.True(invalid.Count == 0, "Namespaces outside the product roots:\n" + string.Join("\n", invalid));
    }

    [Fact]
    public void SharedProjectAndGeneratedTemplatesStayUnderTensorRtSharpRoot()
    {
        string projectPath = Path.Combine(RepositoryPaths.Root, "src", "JYPPX.Shared", "JYPPX.Shared.csproj");
        XDocument project = XDocument.Load(projectPath);
        string? rootNamespace = project.Descendants("RootNamespace").Single().Value;

        Assert.Equal("JYPPX.TensorRtSharp.Shared", rootNamespace);

        string templateRoot = Path.Combine(RepositoryPaths.Root, "native", "templates");
        string templates = string.Join(
            "\n",
            Directory.EnumerateFiles(templateRoot, "*.tpl", SearchOption.TopDirectoryOnly)
                .OrderBy(static path => path, StringComparer.Ordinal)
                .Select(File.ReadAllText));
        Assert.DoesNotContain("JYPPX.Shared.Interop", templates, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.Shared.Generated", templates, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRtSharp.Shared.Interop", templates, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRtSharp.Shared.Generated", templates, StringComparison.Ordinal);
    }

    private static bool IsAllowedProductNamespace(string value)
    {
        return value.Equals("JYPPX.TensorRtSharp", StringComparison.Ordinal) ||
            value.StartsWith("JYPPX.TensorRtSharp.", StringComparison.Ordinal) ||
            value.Equals("JYPPX.CudaSharp", StringComparison.Ordinal) ||
            value.StartsWith("JYPPX.CudaSharp.", StringComparison.Ordinal);
    }

    private static bool IsBuildOutput(string relativePath)
    {
        string normalized = relativePath.Replace('\\', '/');
        return normalized.Contains("/bin/", StringComparison.OrdinalIgnoreCase) ||
            normalized.Contains("/obj/", StringComparison.OrdinalIgnoreCase);
    }
}
