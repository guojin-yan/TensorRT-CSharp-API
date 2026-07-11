using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class DualPackagePublishPreflightMatrixTests
{
    [Fact]
    public void DualPackagePreflightMatrixSeparatesNuGetSmallPackageAndGitHubFullRuntimeWithoutPublishingClaims()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-DualPackagePublishPreflightMatrix.ps1");
        Assert.True(File.Exists(scriptPath), "Dual package publish preflight matrix export script must exist.");

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("nuget-small-bridge-core", script, StringComparison.Ordinal);
        Assert.Contains("github-packages-full-runtime", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("usesPublishToken = $false", script, StringComparison.Ordinal);
        Assert.Contains("requiresOwnerAuthorization = $true", script, StringComparison.Ordinal);
        Assert.Contains("owner-authorization-required", script, StringComparison.Ordinal);
        Assert.Contains("public-package-download-proof-missing", script, StringComparison.Ordinal);
        Assert.Contains("clean-consumer-runtime-proof-missing", script, StringComparison.Ordinal);
        Assert.Contains("post-publish-proof-missing", script, StringComparison.Ordinal);

        RunPowerShell(scriptPath);

        using JsonDocument document = ReadFinalReleaseJson("dual-package-publish-preflight-matrix.json");
        JsonElement root = document.RootElement;

        Assert.Equal("dual-package-publish-preflight-matrix", root.GetProperty("recordKind").GetString());
        Assert.Equal(2, root.GetProperty("routeCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.True(root.GetProperty("requiresOwnerAuthorization").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canPublishGitHubPackages").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canClaimRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
        Assert.Contains("does not publish NuGet", root.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("is not package-consumer runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement[] routes = root.GetProperty("routes").EnumerateArray().ToArray();
        JsonElement nugetRoute = Assert.Single(routes, static item => item.GetProperty("id").GetString() == "nuget-small-bridge-core");
        JsonElement githubRoute = Assert.Single(routes, static item => item.GetProperty("id").GetString() == "github-packages-full-runtime");

        Assert.Equal("nuget.org", nugetRoute.GetProperty("distributionChannel").GetString());
        Assert.Equal("JYPPX.TensorRT.CSharp.API", nugetRoute.GetProperty("packageId").GetString());
        Assert.Contains("no bundled NVIDIA full runtime claim", nugetRoute.GetProperty("packageContents").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("consumer installs CUDA", nugetRoute.GetProperty("dependencyStrategy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(nugetRoute.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(nugetRoute.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
        Assert.DoesNotContain("full runtime package already published", nugetRoute.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Equal("GitHub Packages", githubRoute.GetProperty("distributionChannel").GetString());
        Assert.StartsWith("JYPPX.TensorRT.CSharp.API.runtime.", githubRoute.GetProperty("packageId").GetString(), StringComparison.Ordinal);
        Assert.Contains("heavier native assets", githubRoute.GetProperty("packageContents").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("GitHub Packages", githubRoute.GetProperty("dependencyStrategy").GetString(), StringComparison.Ordinal);
        Assert.False(githubRoute.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(githubRoute.GetProperty("canPublishGitHubPackages").GetBoolean());
        Assert.False(githubRoute.GetProperty("canClaimRuntimeProof").GetBoolean());
        Assert.DoesNotContain("NuGet publication succeeded", githubRoute.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] blockedReasons = root.GetProperty("blockedReasons").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("owner-authorization-required", blockedReasons);
        Assert.Contains("public-package-download-proof-missing", blockedReasons);
        Assert.Contains("clean-consumer-runtime-proof-missing", blockedReasons);
        Assert.Contains("post-publish-proof-missing", blockedReasons);

        foreach (JsonElement route in routes)
        {
            Assert.False(route.GetProperty("performsPublish").GetBoolean());
            Assert.False(route.GetProperty("usesPublishToken").GetBoolean());
            Assert.True(route.GetProperty("requiresOwnerAuthorization").GetBoolean());
            Assert.False(route.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(route.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(route.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(route.GetProperty("isPostPublishProof").GetBoolean());

            JsonElement dryRunRequirement = Assert.Single(
                route.GetProperty("evidenceRequirements").EnumerateArray(),
                static item => item.GetProperty("id").GetString() == "package-dry-run-pack-success");
            Assert.Equal("artifacts/final-release/github-actions-run-evidence-import.json", dryRunRequirement.GetProperty("source").GetString());
        }

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "dual-package-publish-preflight-matrix.md"));
        Assert.Contains("Dual Package Publish Preflight Matrix", markdown, StringComparison.Ordinal);
        Assert.Contains("nuget-small-bridge-core", markdown, StringComparison.Ordinal);
        Assert.Contains("github-packages-full-runtime", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-authorization-required", markdown, StringComparison.Ordinal);
        Assert.Contains("does not publish", markdown, StringComparison.OrdinalIgnoreCase);
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
