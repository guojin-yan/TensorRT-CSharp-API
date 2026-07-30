using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerAndPostPublishProofPlanningTests
{
    [Fact]
    public void PackageConsumerDualRoutePlanDefinesBothPublicConsumptionRoutes()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerDualRouteProofPlan.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("package-consumer-dual-route-proof-plan.json");
        JsonElement root = document.RootElement;

        Assert.Equal("package-consumer-dual-route-proof-plan", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-package-runtime-evidence-required", root.GetProperty("planState").GetString());
        Assert.Equal(2, root.GetProperty("routeCount").GetInt32());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement[] routes = root.GetProperty("routes").EnumerateArray().ToArray();
        Assert.Contains(routes, static route => route.GetProperty("routeId").GetString() == "github-release-managed-plus-bridge-assets");
        Assert.Contains(routes, static route => route.GetProperty("routeId").GetString() == "nuget-managed-plus-bridge-packages");

        foreach (JsonElement route in routes)
        {
            Assert.False(route.GetProperty("performsPublish").GetBoolean());
            Assert.False(route.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(route.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(route.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.False(route.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.True(route.GetProperty("requiredArtifacts").GetArrayLength() >= 6);
            Assert.True(route.GetProperty("requiredHashes").GetArrayLength() >= 4);
            Assert.True(route.GetProperty("requiredEnvironmentMetadata").GetArrayLength() >= 6);
            Assert.Contains("public", route.GetProperty("publicSourceRequirement").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", route.GetProperty("promotionRecordValidatorCommand").GetString(), StringComparison.Ordinal);
        }

        JsonElement githubRoute = routes.Single(static route => route.GetProperty("routeId").GetString() == "github-release-managed-plus-bridge-assets");
        Assert.Contains("Test-PublicReleaseBridgePackageConsumer.ps1", githubRoute.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-RequireReferencedFiles", githubRoute.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotEvidence", githubRoute.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        string raw = root.GetRawText();
        Assert.Contains("Invoke-PublicReleaseBridgePackageConsumer.ps1", raw, StringComparison.Ordinal);
        Assert.Contains("machine-installed", raw, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("same source commit", raw, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("bridge-only", root.GetProperty("requiredDecision").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("github-full-dependency-package", raw, StringComparison.Ordinal);
        Assert.DoesNotContain("runtime dependency bundle", raw, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CleanExternalConsumerExecutionKitKeepsForbiddenSubstitutesOutsideRuntimeEvidence()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerExecutionKit.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("clean-external-consumer-execution-kit.json");
        JsonElement root = document.RootElement;

        Assert.Equal("clean-external-consumer-execution-kit", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-clean-external-consumer-runtime-evidence-required", root.GetProperty("kitState").GetString());
        Assert.Contains("outside the repository", root.GetProperty("cleanRootRequirement").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.True(root.GetProperty("publicPackageSourceRequired").GetBoolean());
        Assert.Equal(0, root.GetProperty("requiredExitCode").GetInt32());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", root.GetProperty("runtimePackageKey").GetString());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string[] disallowed = root.GetProperty("disallowedSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("project-reference substitute", disallowed);
        Assert.Contains("local package-feed substitute", disallowed);
        Assert.Contains("direct nupkg substitute", disallowed);

        JsonElement commands = root.GetProperty("requiredCommands");
        Assert.Contains("dotnet restore", commands.GetProperty("restore").GetString(), StringComparison.Ordinal);
        Assert.Contains("dotnet build", commands.GetProperty("build").GetString(), StringComparison.Ordinal);
        Assert.Contains("dotnet run", commands.GetProperty("smoke").GetString(), StringComparison.Ordinal);

        string[] hashes = root.GetProperty("requiredPackageHashes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("managedPackageSha256", hashes);
        Assert.Contains("nativeBridgeSha256", hashes);
        Assert.Contains("runtimePackageSha256", hashes);
        Assert.Contains("smokeLogSha256", hashes);

        string[] hostMetadata = root.GetProperty("requiredHostMetadata").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("gpuName", hostMetadata);
        Assert.Contains("cudaRuntimeVersion", hostMetadata);
        Assert.Contains("tensorRtVersion", hostMetadata);
        Assert.Contains("cudnnVersion", hostMetadata);
        Assert.True(root.GetProperty("executionSteps").GetArrayLength() >= 6);
    }

    [Fact]
    public void PostPublishVerificationIntakeMapRemainsOwnerInputAndNotProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationIntakeMap.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("post-publish-verification-intake-map.json");
        JsonElement root = document.RootElement;

        Assert.Equal("post-publish-verification-intake-map", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-channel-verification-required", root.GetProperty("intakeState").GetString());
        Assert.True(root.GetProperty("requiredFieldCount").GetInt32() >= 20);
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());

        string raw = root.GetRawText();
        foreach (string required in new[]
        {
            "publicChannelUrl",
            "publishedVersion",
            "packageSource",
            "installCommand",
            "runCommand",
            "smokeLogPath",
            "smokeLogSha256",
            "stdoutSummary",
            "stderrSummary",
            "ownerReviewer",
            "reviewedAtUtc",
            "rollbackOrDeprecationPlanReference"
        })
        {
            Assert.Contains(required, raw, StringComparison.Ordinal);
        }

        Assert.Contains("Test-PostPublishVerificationRecord.ps1", root.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("not post-publish proof", root.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalActionMapLinksPackageConsumerAndPostPublishPlanningArtifacts()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerDualRouteProofPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerExecutionKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationIntakeMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("final-publish-action-required-evidence-map.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-publish-action-required-evidence-map", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-evidence-required", root.GetProperty("mapState").GetString());
        Assert.Equal(2, root.GetProperty("packageConsumerRouteCount").GetInt32());
        Assert.True(root.GetProperty("cleanExternalConsumerStepCount").GetInt32() >= 6);
        Assert.True(root.GetProperty("postPublishRequiredFieldCount").GetInt32() >= 20);
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement[] actions = root.GetProperty("actions").EnumerateArray().ToArray();
        JsonElement packageAction = actions.Single(static action => action.GetProperty("id").GetString() == "package-consumer-runtime-owner-proof-required");
        JsonElement postPublishAction = actions.Single(static action => action.GetProperty("id").GetString() == "post-publish-verification-owner-proof-required");

        Assert.Equal("artifacts/final-release/package-consumer-dual-route-proof-plan.json", packageAction.GetProperty("planningArtifact").GetString());
        Assert.Equal("artifacts/final-release/clean-external-consumer-execution-kit.json", packageAction.GetProperty("executionKit").GetString());
        Assert.Contains("github-release-managed-plus-bridge-assets", packageAction.GetProperty("routeIds").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("nuget-managed-plus-bridge-packages", packageAction.GetProperty("routeIds").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("public package source", packageAction.GetProperty("publicSourceRequirement").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Equal("artifacts/final-release/post-publish-verification-intake-map.json", postPublishAction.GetProperty("intakeMapArtifact").GetString());
        Assert.Contains("public channel", postPublishAction.GetProperty("publicSourceRequirement").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.False(packageAction.GetProperty("performsPublish").GetBoolean());
        Assert.False(packageAction.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(packageAction.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(packageAction.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(postPublishAction.GetProperty("performsPublish").GetBoolean());
        Assert.False(postPublishAction.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(postPublishAction.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(postPublishAction.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    [Fact]
    public void PublicDocsGateStillHasNoBlockedOverclaimMatches()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("public-docs-package-metadata-gate.json");
        JsonElement root = document.RootElement;

        Assert.Equal("public-docs-package-metadata-gate", root.GetProperty("recordKind").GetString());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, root.GetProperty("blockedMatchCount").GetInt32());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalsePublishAndCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
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
