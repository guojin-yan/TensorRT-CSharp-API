using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimePackageSmokeCommandTemplateTests
{
    [Fact]
    public void RuntimePackageSmokeTemplateDocumentsCommandsAndForbiddenSubstitutes()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-package-smoke-command-template.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("runtime-package-smoke-command-template.v1", root.GetProperty("schemaVersion").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        foreach (string term in new[]
                 {
                     "windowsPowerShell",
                     "linuxBash",
                     "restore",
                     "build",
                     "nativeLoad",
                     "deserialize",
                     "sampleRun",
                     "validator",
                     "hostMetadata",
                     "packageSource",
                     "managedPackageSha256",
                     "runtimePackageSha256",
                     "runtime proof",
                     "build-only",
                     "dry-run",
                     "template",
                     "local feed",
                     "ProjectReference",
                     "direct `.nupkg`",
                     "TensorRtExec report",
                     "YoloVision matrix",
                     "OnnxToEngine report",
                     "readonly diagnostics",
                 })
        {
            Assert.Contains(term, text, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("dotnet nuget push", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
    }
}
