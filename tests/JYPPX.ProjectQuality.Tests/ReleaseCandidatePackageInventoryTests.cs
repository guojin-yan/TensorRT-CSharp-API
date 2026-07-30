using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidatePackageInventoryTests
{
    [Fact]
    public void ExportScriptDefinesPackageInventoryAndProofBoundaries()
    {
        string script = ReadSource("eng", "Export-ReleaseCandidatePackageInventory.ps1");
        string finalReviewScript = ReadSource("eng", "Export-FinalPackageReviewBundle.ps1");
        string releaseProofScript = ReadSource("eng", "Export-ReleasePackageProofBundle.ps1");
        string plan = File.ReadAllText(Path.Combine(
            Directory.GetParent(RepositoryPaths.Root)!.FullName,
            "plan",
            "2026-07-05-0743-发布候选包完整门禁-阶段计划.md"));
        string localFeedDoc = ReadSource("docs", "articles", "zh-cn", "local-nuget-feed-consumer.md");

        Assert.Contains("recordKind = \"release-candidate-package-inventory\"", script, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory.json", script, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory.md", script, StringComparison.Ordinal);
        Assert.Contains("Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256", script, StringComparison.Ordinal);
        Assert.Contains("role = $role", script, StringComparison.Ordinal);
        Assert.Contains("\"split-bridge\"", script, StringComparison.Ordinal);
        Assert.Contains("\"split-cuda-cudnn\"", script, StringComparison.Ordinal);
        Assert.Contains("\"split-tensorrt\"", script, StringComparison.Ordinal);
        Assert.Contains("\"split-meta\"", script, StringComparison.Ordinal);
        Assert.Contains("splitBridgePackageReady", script, StringComparison.Ordinal);
        Assert.Contains("packageSetReady", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canUseAsPublicPackageProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("Local package inventory accepts only managed and bridge candidates", script, StringComparison.Ordinal);
        Assert.Contains("retiredPackageCandidateCount", script, StringComparison.Ordinal);
        Assert.Contains("fullRuntimePackageRequired = $false", script, StringComparison.Ordinal);
        Assert.Contains("vendorRuntimePackagesForbidden = $true", script, StringComparison.Ordinal);
        Assert.Contains("B-tier safe alternative proof is not runtime proof", script, StringComparison.Ordinal);
        Assert.Contains("CompatibleBridgeRuntimePackageKey", script, StringComparison.Ordinal);
        Assert.Contains("compatibleBridgeRuntimeProofReady", script, StringComparison.Ordinal);
        Assert.Contains("compatible-host-bridge-package-runtime", script, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory.json", finalReviewScript, StringComparison.Ordinal);
        Assert.Contains("packageInventorySha256Ready", finalReviewScript, StringComparison.Ordinal);
        Assert.Contains("compatibleBridgeRuntimeProofReady", finalReviewScript, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory.json", releaseProofScript, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory", releaseProofScript, StringComparison.Ordinal);
        Assert.Contains("packageInventoryReady", releaseProofScript, StringComparison.Ordinal);

        Assert.Contains("release-candidate-package-inventory.json", plan, StringComparison.Ordinal);
        Assert.Contains("local-feed-is-not-post-publish-proof", localFeedDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void ExportedInventoryAcceptsManagedAndBridgeAndFlagsRetiredCandidates()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidatePackageInventory.ps1"));

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-package-inventory.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-package-inventory.md");
        Assert.True(File.Exists(jsonPath), "Inventory JSON should exist.");
        Assert.True(File.Exists(markdownPath), "Inventory markdown should exist.");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;
        Assert.Equal("release-candidate-package-inventory", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(root.GetProperty("managedPackageReady").GetBoolean());
        Assert.False(root.GetProperty("fullRuntimePackageReady").GetBoolean());
        Assert.False(root.GetProperty("fullRuntimePackageRequired").GetBoolean());
        Assert.True(root.GetProperty("vendorRuntimePackagesForbidden").GetBoolean());
        Assert.True(root.GetProperty("splitBridgePackageReady").GetBoolean());
        Assert.True(root.GetProperty("splitRuntimePackagesReady").GetBoolean());
        Assert.True(root.GetProperty("sha256Ready").GetBoolean());
        int retiredCandidateCount = root.GetProperty("retiredPackageCandidateCount").GetInt32();
        Assert.Equal(retiredCandidateCount == 0, root.GetProperty("packageSetReady").GetBoolean());
        Assert.True(root.GetProperty("compatibleBridgeRuntimeProofReady").GetBoolean());
        Assert.True(root.GetProperty("compatibleBridgePackageCount").GetInt32() >= 1);
        Assert.Empty(root.GetProperty("missingSplitRoles").EnumerateArray());
        Assert.Contains("not public channel proof", root.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement packages = root.GetProperty("packages");
        Assert.Contains(packages.EnumerateArray(), static package =>
            package.GetProperty("role").GetString() == "managed" &&
            package.GetProperty("packageId").GetString() == "JYPPX.TensorRT.CSharp.API" &&
            package.GetProperty("version").GetString() == "4.0.0");
        Assert.Contains(packages.EnumerateArray(), static package => package.GetProperty("role").GetString() == "split-bridge");
        Assert.Equal(retiredCandidateCount, packages.EnumerateArray().Count(static package =>
            package.GetProperty("role").GetString() is not ("managed" or "split-bridge")));
        Assert.Contains(packages.EnumerateArray(), static package =>
            package.GetProperty("role").GetString() == "split-bridge" &&
            package.GetProperty("runtimePackageKey").GetString() == "win-x64-trt10.11-cuda12.9-cudnn9.22");
        Assert.Contains(root.GetProperty("compatibleBridgeRuntimeProofs").EnumerateArray(), static proof =>
            proof.GetProperty("runtimePackageKey").GetString() == "win-x64-trt10.11-cuda12.9-cudnn9.22" &&
            proof.GetProperty("ready").GetBoolean() &&
            proof.GetProperty("proofClassification").GetString() == "compatible-host-bridge-package-runtime");

        foreach (JsonElement package in packages.EnumerateArray())
        {
            Assert.Matches("^[a-f0-9]{64}$", package.GetProperty("sha256").GetString()!);
            Assert.True(package.GetProperty("sizeBytes").GetInt64() > 0);
            Assert.False(string.IsNullOrWhiteSpace(package.GetProperty("relativePath").GetString()));
        }

        string markdown = File.ReadAllText(markdownPath);
        Assert.Contains("Release Candidate Package Inventory", markdown, StringComparison.Ordinal);
        Assert.Contains("full/vendor runtime packages required: `False`", markdown, StringComparison.Ordinal);
        Assert.Contains("retired package candidates found", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseBundlesConsumePackageInventoryWithoutPromotingItToPublishProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidatePackageInventory.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));

        using JsonDocument proof = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-package-proof-bundle.json")));
        JsonElement proofRoot = proof.RootElement;
        Assert.Equal("release-candidate-package-inventory", proofRoot.GetProperty("packageInventoryState").GetString());
        Assert.True(proofRoot.GetProperty("packageInventorySplitBridgePackageReady").GetBoolean());
        Assert.Equal(
            proofRoot.GetProperty("packageInventoryRetiredPackageCandidateCount").GetInt32() == 0,
            proofRoot.GetProperty("packageInventoryReady").GetBoolean());
        Assert.True(proofRoot.GetProperty("packageInventorySha256Ready").GetBoolean());
        Assert.False(proofRoot.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(proofRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        bool inventoryReady = proofRoot.GetProperty("packageInventoryRetiredPackageCandidateCount").GetInt32() == 0;
        Assert.Contains(proofRoot.GetProperty("evidenceItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "release-candidate-package-inventory" &&
            item.GetProperty("passed").GetBoolean() == inventoryReady &&
            item.GetProperty("state").GetString()!.Contains("bridgeReady=True", StringComparison.Ordinal) &&
            item.GetProperty("state").GetString()!.Contains("bridgeOnlySetReady=True", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("only managed and bridge candidates", StringComparison.OrdinalIgnoreCase));

        using JsonDocument finalReview = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "final-package-review-bundle.json")));
        JsonElement finalRoot = finalReview.RootElement;
        Assert.Equal("release-candidate-package-inventory", finalRoot.GetProperty("packageInventoryState").GetString());
        Assert.True(finalRoot.GetProperty("packageInventorySplitBridgePackageReady").GetBoolean());
        Assert.True(finalRoot.GetProperty("packageInventorySplitRuntimePackagesReady").GetBoolean());
        Assert.False(finalRoot.GetProperty("packageInventoryFullRuntimePackageRequired").GetBoolean());
        Assert.True(finalRoot.GetProperty("packageInventorySha256Ready").GetBoolean());
        Assert.True(finalRoot.GetProperty("compatibleBridgeRuntimeProofReady").GetBoolean());
        Assert.True(finalRoot.GetProperty("compatibleBridgePackageCount").GetInt32() >= 1);
        Assert.False(finalRoot.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(finalRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(finalRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-candidate-package-inventory.json");
        Assert.Contains(finalRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/package-consumer/bridge-runtime/win-x64-trt10.11-cuda12.9-cudnn9.22/bridge-package-runtime-consumer-proof.json");
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static string RunPowerShell(string scriptPath)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "pwsh",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(output + Environment.NewLine + error);
        }

        return output;
    }
}
