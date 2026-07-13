using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicApiHandleExposureAuditTests
{
    [Fact]
    public void HighLevelPublicApiDoesNotExposePluginCreatorBorrowedPointers()
    {
        string root = Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp");
        string[] files = Directory.GetFiles(root, "*.cs", SearchOption.AllDirectories)
            .Where(static path =>
                !path.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) &&
                !path.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase))
            .ToArray();

        Assert.NotEmpty(files);

        var offenders = new List<string>();
        var publicPointerPattern = new Regex(@"\bpublic\b[^\r\n;{}=]*(?:IntPtr|UIntPtr|nint|nuint)\b[^\r\n;{}=]*(?:Plugin|Creator|Field|Borrowed|DebugTensor|OutputBuffer|DevicePointer)", RegexOptions.Compiled);

        foreach (string file in files)
        {
            string text = File.ReadAllText(file);
            if (!text.Contains("public", StringComparison.Ordinal))
            {
                continue;
            }

            foreach (Match match in publicPointerPattern.Matches(text))
            {
                string relative = Path.GetRelativePath(RepositoryPaths.Root, file);
                offenders.Add(relative + ": " + match.Value.Trim());
            }
        }

        Assert.DoesNotContain(offenders, static offender =>
            offender.Contains("Plugin", StringComparison.OrdinalIgnoreCase) ||
            offender.Contains("Creator", StringComparison.OrdinalIgnoreCase) ||
            offender.Contains("Borrowed", StringComparison.OrdinalIgnoreCase) ||
            offender.Contains("DebugTensor", StringComparison.OrdinalIgnoreCase) ||
            offender.Contains("OutputBuffer", StringComparison.OrdinalIgnoreCase) ||
            offender.Contains("DevicePointer", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void PluginInventoryArticlesAndWrappersDocumentPointerFreeSurface()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "csharp-public-api-handle-exposure-audit.md"));

        string inventoryArticle = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "plugin-inventory-field-metadata-smoke-guide.md"));

        string inventoryWrapper = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "TensorRtPluginRegistryInventory.cs"));

        Assert.Contains("不暴露裸 `IntPtr`", article, StringComparison.Ordinal);
        Assert.Contains("pointer-free", inventoryArticle, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("public IntPtr", inventoryWrapper, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", inventoryWrapper, StringComparison.Ordinal);
    }
}
