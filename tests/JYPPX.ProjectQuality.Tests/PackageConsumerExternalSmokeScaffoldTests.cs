using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerExternalSmokeScaffoldTests
{
    [Fact]
    public void PackageConsumerExternalSmokeScaffoldExportsCleanConsumerShapeButNotProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "New-PackageConsumerExternalSmokeScaffold.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument scaffoldDocument = ReadFinalReleaseJson("package-consumer-external-smoke-scaffold.json");
        JsonElement scaffold = scaffoldDocument.RootElement;
        Assert.Equal("package-consumer-external-smoke-scaffold", scaffold.GetProperty("recordKind").GetString());
        Assert.True(scaffold.GetProperty("outputRootIsOutsideRepository").GetBoolean());
        Assert.False(scaffold.GetProperty("isProof").GetBoolean());
        Assert.False(scaffold.GetProperty("performsPublish").GetBoolean());
        Assert.False(scaffold.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(scaffold.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement scan = scaffold.GetProperty("scan");
        Assert.Equal(0, scan.GetProperty("projectReferenceCount").GetInt32());
        Assert.False(scan.GetProperty("usesDirectNupkg").GetBoolean());
        Assert.False(scan.GetProperty("canBePublicProof").GetBoolean());

        string projectPath = scaffold.GetProperty("projectPath").GetString()!;
        string projectText = File.ReadAllText(projectPath);
        Assert.DoesNotContain("<ProjectReference", projectText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".nupkg", projectText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("PackageReference", projectText, StringComparison.Ordinal);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("package-consumer-external-smoke-scaffold", evidence.GetProperty("packageConsumerExternalSmokeScaffoldState").GetString());
        Assert.True(evidence.GetProperty("packageConsumerExternalSmokeScaffoldOutputRootIsOutsideRepository").GetBoolean());
        Assert.True(evidence.GetProperty("packageConsumerExternalSmokeScaffoldPublicPackageSourceIsLocal").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerExternalSmokeScaffoldCanBePublicProof").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerExternalSmokeScaffoldIsProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-external-smoke-scaffold");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not runtime proof", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-external-smoke-scaffold.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-external-smoke-scaffold.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-external-smoke-scaffold.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/package-consumer-external-smoke-scaffold.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/package-consumer-external-smoke-scaffold.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("不是运行 proof", article, StringComparison.Ordinal);
        Assert.Contains("package consumer external smoke scaffold: `package-consumer-external-smoke-scaffold`", evidenceMarkdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
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
