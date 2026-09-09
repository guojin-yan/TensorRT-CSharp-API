using System.Text.Json;
using System.Diagnostics;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseAutomationTests
{
    private static readonly string[] WindowsRuntimeKeys =
    [
        "win-x64-trt8.6-cuda11.8-cudnn8.9",
        "win-x64-trt8.6-cuda12.1-cudnn8.9",
        "win-x64-trt10.11-cuda11.8-cudnn8.9",
        "win-x64-trt10.11-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda13.2-cudnn9.22",
    ];

    [Fact]
    public void RuntimeWorkflowsDefaultToCompleteHostedMatrices()
    {
        string releaseBundle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));
        string runtimeWindows = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-windows.yml"));
        string runtimeLinux = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml"));

        foreach (string key in WindowsRuntimeKeys)
        {
            Assert.Contains(key, releaseBundle, StringComparison.Ordinal);
            Assert.Contains(key, runtimeWindows, StringComparison.Ordinal);
        }

        Assert.Contains("git config --global http.version HTTP/1.1", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("GITHUB_PATH", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("$ghVersion = '2.77.0'", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("gh_${ghVersion}_windows_amd64.zip", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("$ghBin = Join-Path $ghRoot 'bin'", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("gh.exe", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("JYPPX_LOCAL_SOURCE_ROOT", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("Local source HEAD '$sourceCommit' does not match workflow commit '$expectedCommit'", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("steps.checkout-mode.outputs.use_local_source != 'true'", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("default: hosted-all", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("default: auto", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("runtime_key_set", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("runner_mode", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("fromJson(matrix.runsOnJson)", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("publish_runtime_to_nuget", releaseBundle, StringComparison.Ordinal);
        Assert.DoesNotContain("publish_to_nuget=$PUBLISH_RUNTIME_TO_NUGET", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("-f \"publish_to_nuget=false\"", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("Publish exact package set to nuget.org", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("runtime_delivery_mode == 'split'", runtimeWindows, StringComparison.Ordinal);
        Assert.Contains("runtime_delivery_mode == 'split'", runtimeLinux, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseNotesKeepReadmeSummaryVersionIndexAndDetailedRecordAligned()
    {
        string english = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string chinese = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string index = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "releases", "README.md"));
        string details = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "releases", "4.0.0.md"));

        Assert.Contains("Latest Update: 4.0.0", english, StringComparison.Ordinal);
        Assert.Contains("本次更新：4.0.0", chinese, StringComparison.Ordinal);
        Assert.Contains("docs/releases/4.0.0.md", english, StringComparison.Ordinal);
        Assert.Contains("docs/releases/4.0.0.md", chinese, StringComparison.Ordinal);
        Assert.Contains("[4.0.0](4.0.0.md)", index, StringComparison.Ordinal);
        Assert.Contains("正式公开包集合固定为 19 个包", details, StringComparison.Ordinal);
        Assert.Contains("## 兼容性与环境要求", details, StringComparison.Ordinal);
        Assert.Contains("## 验证范围与已知限制", details, StringComparison.Ordinal);
        Assert.Contains("不包含 CUDA、cuDNN、TensorRT 或 NVRTC", details, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimePublicationRequiresExplicitOwnerApprovalAndNuGetSecretPreflight()
    {
        string releaseBundle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));
        string runtimeWindows = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-windows.yml"));
        string runtimeLinux = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml"));

        Assert.Contains("'${{ inputs.publish_runtime_to_nuget }}' -eq 'true'", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("[ \"$PUBLISH_RUNTIME_TO_NUGET\" = \"true\" ]", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("Publishing managed or Bridge packages to nuget.org requires the repository secret NUGET_API_KEY", releaseBundle, StringComparison.Ordinal);
        Assert.Equal(
            6,
            System.Text.RegularExpressions.Regex.Matches(
                releaseBundle,
                "owner_publish_approved=\\$OWNER_PUBLISH_APPROVED").Count);
        Assert.Contains("nuget.org publication must include both the managed package and all Bridge packages", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("GitHub Packages publication must include both the managed package and all Bridge packages", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("Formal package publication requires all 6 Windows, 9 hosted Linux, and 3 Ubuntu 20.04 hosted-container Bridge lanes", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("publish_to_github_packages=false", releaseBundle, StringComparison.Ordinal);
        Assert.DoesNotContain("Publish-ReleaseNuGetAssetsToGitHubPackages.ps1", releaseBundle, StringComparison.Ordinal);

        foreach (string workflow in new[] { runtimeWindows, runtimeLinux })
        {
            Assert.Matches("owner_publish_approved:[\\s\\S]*?default: false", workflow);
            Assert.Contains("Package or Release publication requires owner_publish_approved=true", workflow, StringComparison.Ordinal);
            Assert.Contains("inputs.publish_to_nuget", workflow, StringComparison.Ordinal);
            Assert.Contains("inputs.publish_to_github_packages", workflow, StringComparison.Ordinal);
            Assert.Contains("publish-runtime:", workflow, StringComparison.Ordinal);
            Assert.Matches("attach_to_github_release:[\\s\\S]*?default: false", workflow);
        }
    }

    [Fact]
    public void PublicationJobsRequireProductionReleaseEnvironment()
    {
        string packageManaged = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "package-managed.yml"));
        string packageSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "package-source.yml"));
        string releaseBundle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));
        string docsRelease = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "docs-release.yml"));
        string[] runtimeWorkflows =
        [
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-windows.yml")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml")),
        ];

        Assert.True(System.Text.RegularExpressions.Regex.Matches(packageManaged, "name: production-release").Count == 3);
        Assert.True(System.Text.RegularExpressions.Regex.Matches(packageSource, "name: production-release").Count == 1);
        Assert.True(System.Text.RegularExpressions.Regex.Matches(releaseBundle, "name: production-release").Count == 1);
        Assert.True(System.Text.RegularExpressions.Regex.Matches(docsRelease, "name: production-release").Count == 1);
        Assert.DoesNotContain(
            "name: production-release",
            packageManaged[..packageManaged.IndexOf("  attach-github-release:", StringComparison.Ordinal)],
            StringComparison.Ordinal);
        Assert.Equal(9, System.Text.RegularExpressions.Regex.Matches(packageManaged, "PRODUCTION_RELEASE_READY").Count);
        Assert.Contains("Test-ProductionReleaseEnvironment.ps1", packageSource, StringComparison.Ordinal);
        Assert.Contains("PRODUCTION_RELEASE_READY", packageSource, StringComparison.Ordinal);
        Assert.Contains("Test-ProductionReleaseEnvironment.ps1", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("PRODUCTION_RELEASE_READY", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("inputs.owner_publish_approved && github.repository_owner == 'guojin-yan'", docsRelease, StringComparison.Ordinal);
        Assert.Contains("Test-ProductionReleaseEnvironment.ps1", docsRelease, StringComparison.Ordinal);
        Assert.Contains("PRODUCTION_RELEASE_READY", docsRelease, StringComparison.Ordinal);
        Assert.Contains("actions/deploy-pages", docsRelease, StringComparison.Ordinal);

        foreach (string workflow in runtimeWorkflows)
        {
            int publishJobIndex = workflow.IndexOf("  publish-runtime:", StringComparison.Ordinal);
            Assert.True(publishJobIndex > 0);
            Assert.DoesNotContain("name: production-release", workflow[..publishJobIndex], StringComparison.Ordinal);
            Assert.Contains("needs: pack-runtime", workflow[publishJobIndex..], StringComparison.Ordinal);
            Assert.Contains("name: production-release", workflow[publishJobIndex..], StringComparison.Ordinal);
            Assert.Contains("inputs.publish_to_nuget || inputs.publish_to_github_packages", workflow[publishJobIndex..], StringComparison.Ordinal);
            Assert.DoesNotContain("secrets.NUGET_API_KEY", workflow[..publishJobIndex], StringComparison.Ordinal);
            Assert.Contains("Test-ProductionReleaseEnvironment.ps1", workflow[publishJobIndex..], StringComparison.Ordinal);
            Assert.Contains("PRODUCTION_RELEASE_READY", workflow[publishJobIndex..], StringComparison.Ordinal);
        }

        string environmentGate = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-ProductionReleaseEnvironment.ps1"));
        Assert.Contains("PRODUCTION_RELEASE_READY", environmentGate, StringComparison.Ordinal);
        Assert.Contains("production-release Environment is not configured", environmentGate, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseBundlePublishesOneValidatedStablePayloadFromTheCurrentCommit()
    {
        string workflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));

        Assert.Contains("Formal release payload must contain exactly 19 NuGet packages: 1 managed package and 18 Bridge packages", workflow, StringComparison.Ordinal);
        Assert.Contains("fixed 6 Windows + 12 Linux runtime manifest", workflow, StringComparison.Ordinal);
        Assert.Contains("TensorRtSharp4.0-source-$RELEASE_VERSION.zip", workflow, StringComparison.Ordinal);
        Assert.Contains(".commit == $commit", workflow, StringComparison.Ordinal);
        Assert.Contains(".sha256 == $sha", workflow, StringComparison.Ordinal);
        Assert.Contains("title=\"TensorRT-CSharp-API $RELEASE_VERSION\"", workflow, StringComparison.Ordinal);
        Assert.Contains("notes_path=\"$GITHUB_WORKSPACE/docs/releases/$RELEASE_VERSION.md\"", workflow, StringComparison.Ordinal);
        Assert.Contains("repos/$GH_REPO/git/refs", workflow, StringComparison.Ordinal);
        Assert.Contains("ref=refs/tags/$tag", workflow, StringComparison.Ordinal);
        Assert.Contains("sha=$GH_HEAD_SHA", workflow, StringComparison.Ordinal);
        Assert.Contains("if tag_commit=\"$(gh api", workflow, StringComparison.Ordinal);
        Assert.Contains("--draft", workflow, StringComparison.Ordinal);
        Assert.Contains("draft:false", workflow, StringComparison.Ordinal);
        Assert.Contains("prerelease:false", workflow, StringComparison.Ordinal);
        Assert.Contains("gh api --method PATCH", workflow, StringComparison.Ordinal);
        Assert.Contains("Stable GitHub Release requires exactly 20 assets", workflow, StringComparison.Ordinal);
        Assert.Contains("tag_commit", workflow, StringComparison.Ordinal);
        Assert.Contains("gh workflow run docs-release.yml", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeWorkflowsUseBoundedFirstReleasePackageContractTests()
    {
        string[] workflows =
        [
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-windows.yml")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml")),
        ];
        string[] requiredTestClasses =
        [
            "RuntimeManifestTests",
            "ManagedPackageTests",
            "ExternalVendorRuntimePackagePolicyTests",
            "BridgePackageConsumerTests",
            "YoloVisionManagedPackagePublicationTests",
            "ReleaseAutomationTests",
            "ReleaseQualityGateWorkflowTests",
        ];

        foreach (string workflow in workflows)
        {
            Assert.Contains("Test first release core package contracts", workflow, StringComparison.Ordinal);
            Assert.Contains("--no-build --filter", workflow, StringComparison.Ordinal);
            Assert.Contains("publish_to_nuget", workflow, StringComparison.Ordinal);
            Assert.Contains("Publish", workflow, StringComparison.Ordinal);
            Assert.Contains("api.nuget.org/v3/index.json", workflow, StringComparison.Ordinal);
            Assert.Contains("Push-NuGetPackages.ps1", workflow, StringComparison.Ordinal);
            Assert.DoesNotContain("--no-build\n", workflow.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);
            foreach (string testClass in requiredTestClasses)
            {
                Assert.Contains($"FullyQualifiedName~{testClass}", workflow, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void RootReadmesDescribeBridgeOnlyRuntimePublication()
    {
        string english = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string chinese = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string samples = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));

        Assert.Contains("NVIDIA runtime redistribution is retired", english, StringComparison.Ordinal);
        Assert.Contains("`.Bridge` packages with `split_package_roles=bridge`", english, StringComparison.Ordinal);
        Assert.Contains("windows_split_package_roles=bridge", chinese, StringComparison.Ordinal);
        Assert.Contains("不重新分发 CUDA、cuDNN、TensorRT 厂商运行库", chinese, StringComparison.Ordinal);
        Assert.DoesNotContain("windows_split_package_roles=bridge,collection", chinese, StringComparison.Ordinal);
        Assert.DoesNotContain("windows_cuda_cudnn_package_version", chinese, StringComparison.Ordinal);
        Assert.DoesNotContain("windows_tensorrt_package_version", chinese, StringComparison.Ordinal);
        Assert.Contains("dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version 4.0.0", samples, StringComparison.Ordinal);
        Assert.Contains("img.shields.io/nuget/v/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge.svg?label=version", english, StringComparison.Ordinal);
        Assert.Contains("img.shields.io/nuget/v/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge.svg?label=version", chinese, StringComparison.Ordinal);
        Assert.Contains("NuGet.org", english, StringComparison.Ordinal);
        Assert.Contains("NuGet.org", chinese, StringComparison.Ordinal);
    }

    [Fact]
    public void LinuxTargetCatalogKeepsUbuntuAndFutureArchitectureLinesExplicit()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "linux-runtime-targets.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        JsonElement[] targets = document.RootElement.GetProperty("targets").EnumerateArray().ToArray();
        Assert.Contains(targets, static target => target.GetProperty("target").GetString() == "ubuntu22.04-x64-hosted");
        Assert.Contains(targets, static target => target.GetProperty("target").GetString() == "ubuntu24.04-x64-hosted");
        Assert.Contains(targets, static target => target.GetProperty("target").GetString() == "ubuntu20.04-x64-hosted-container");

        JsonElement[] futureTargets = document.RootElement.GetProperty("futureTargets").EnumerateArray().ToArray();
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "linux-arm64-sbsa");
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "linux-jetson-l4t");
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "non-ubuntu-linux");
    }

    [Fact]
    public void ReleasePublicationAuditCanSkipPrerequisiteNoiseSeparatelyFromInventory()
    {
        string workflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-publication-audit.yml"));
        string publicationState = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleasePublicationState.ps1"));

        Assert.Contains("check_remote_release_prerequisites", workflow, StringComparison.Ordinal);
        Assert.Contains("check_runner_availability", workflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ inputs.check_remote_release_prerequisites }}", workflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ inputs.check_runner_availability }}", workflow, StringComparison.Ordinal);
        Assert.Contains("include_release_readiness", workflow, StringComparison.Ordinal);
        Assert.Contains("managed_version", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.TensorRT.CSharp.API.Classification", workflow, StringComparison.Ordinal);
        Assert.Contains("managedExtensionPackageId =", publicationState, StringComparison.Ordinal);
        Assert.Contains("managedExtensionPackageIds =", publicationState, StringComparison.Ordinal);
    }

    [Fact]
    public void NuGetPushScriptFailsFastForNonRetryableNuGetOrgAuthorizationErrors()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Push-NuGetPackages.ps1"));

        Assert.Contains("Test-IsNonRetryableNuGetAuthorizationFailure", script, StringComparison.Ordinal);
        Assert.Contains("nuget.org rejected the package with a non-retryable authentication/authorization failure", script, StringComparison.Ordinal);
        Assert.Contains("does not have permission", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NuGetOrgPublicationDocsRequirePackageScopedApiKey()
    {
        string englishReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string chineseReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string englishGate = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "en", "release-candidate-gate.md"));
        string chineseGate = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-candidate-gate.md"));
        string summaryScript = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseGateSummary.ps1"));

        Assert.Contains("core package permission plus package-scoped push permission", englishReadme, StringComparison.Ordinal);
        Assert.Contains("each project-owned `.Bridge` ID", englishReadme, StringComparison.Ordinal);
        Assert.Contains("nuget.org `403`", englishReadme, StringComparison.Ordinal);
        Assert.Contains("`.Bridge` ID", chineseReadme, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API.YoloVision", chineseReadme, StringComparison.Ordinal);
        Assert.Contains("严禁上传", chineseReadme, StringComparison.Ordinal);
        Assert.Contains("nuget.org `403`", chineseReadme, StringComparison.Ordinal);
        Assert.Contains("push permission for the managed package ID", englishGate, StringComparison.Ordinal);
        Assert.Contains("nuget.org `403`", chineseGate, StringComparison.Ordinal);
        Assert.Contains("NUGET_API_KEY` must be an active plain-text nuget.org key", summaryScript, StringComparison.Ordinal);
    }

    [Fact]
    public void RemoteReleaseBundleDryRunCarriesBridgeOnlyPolicy()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Invoke-RemoteReleaseBundle.ps1");
        string output = RunPowerShell(
            script,
            "-Version", "4.0.7000",
            "-RuntimeVersion", "4.0.7000",
            "-RunWindowsRuntimePackaging",
            "-WindowsSplitPackageRoles", "bridge",
            "-RunLinuxRuntimePackaging",
            "-LinuxRuntimeKeySet", "hosted-all",
            "-LinuxSplitPackageRoles", "bridge",
            "-OwnerPublishApproved", "true",
            "-PublishRuntimeToGitHubPackages", "true",
            "-DryRun");

        Assert.Contains("gh workflow run release-bundle.yml", output, StringComparison.Ordinal);
        Assert.Contains("release_config_json=", output, StringComparison.Ordinal);
        Assert.Contains("owner_publish_approved=true", output, StringComparison.Ordinal);
        Assert.DoesNotContain("cuda_cudnn_package_version", output, StringComparison.Ordinal);
        Assert.DoesNotContain("tensorrt_package_version", output, StringComparison.Ordinal);
        Assert.Contains("hosted-all", output, StringComparison.Ordinal);
        Assert.Contains("windows_split_package_roles=bridge", output, StringComparison.Ordinal);
        Assert.Contains("linux_split_package_roles=bridge", output, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("hosted-all", "hosted", 9, "linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9", "linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22")]
    [InlineData("ubuntu24-hosted", "hosted", 3, "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22", "linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22")]
    [InlineData("hosted-container-ubuntu20", "hosted-container", 3, "linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9", "linux-x64-ubuntu20.04-trt10.11-cuda11.8-cudnn8.9")]
    public void LinuxRuntimeKeySetsResolveExpectedPackageLines(
        string keySet,
        string runnerMode,
        int expectedCount,
        string expectedFirstKey,
        string expectedSecondKey)
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeKeySet.ps1");
        string output = RunPowerShell(
            script,
            "-Platform", "linux",
            "-RuntimeKeySet", keySet,
            "-RunnerMode", runnerMode,
            "-OutputFormat", "json");

        string[] keys = JsonSerializer.Deserialize<string[]>(output)!;

        Assert.Equal(expectedCount, keys.Length);
        Assert.Contains(expectedFirstKey, keys);
        Assert.Contains(expectedSecondKey, keys);
    }

    [Fact]
    public void LegacyUbuntu20SelfHostedKeySetFailsWithMigrationGuidance()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeKeySet.ps1");
        (int exitCode, string output) = RunPowerShellAllowFailure(
            script,
            "-Platform", "linux",
            "-RuntimeKeySet", "self-hosted-ubuntu20",
            "-RunnerMode", "self-hosted",
            "-OutputFormat", "json");

        Assert.NotEqual(0, exitCode);
        Assert.Contains("hosted-container-ubuntu20", output, StringComparison.Ordinal);
        Assert.Contains("runner_mode='hosted-container'", output, StringComparison.Ordinal);
    }

    [Fact]
    public void Ubuntu20HostedContainerMatrixUsesOfficialAptDependencyPreparation()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeMatrix.ps1");
        string output = RunPowerShell(
            script,
            "-Platform", "linux",
            "-RuntimeKey", "linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9",
            "-RunnerMode", "hosted-container");

        using JsonDocument document = JsonDocument.Parse(output);
        JsonElement entry = Assert.Single(document.RootElement.EnumerateArray());

        Assert.Equal("hosted-container", entry.GetProperty("runnerMode").GetString());
        Assert.Equal("apt", entry.GetProperty("nvidiaDependencyMode").GetString());
        Assert.Equal("ubuntu:20.04", entry.GetProperty("containerImage").GetString());
        Assert.Contains("ubuntu-latest", entry.GetProperty("runsOnJson").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Ubuntu22HostedMatrixUsesTheMatchingJobContainer()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeMatrix.ps1");
        string output = RunPowerShell(
            script,
            "-Platform", "linux",
            "-RuntimeKey", "linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22",
            "-RunnerMode", "hosted");

        using JsonDocument document = JsonDocument.Parse(output);
        JsonElement entry = Assert.Single(document.RootElement.EnumerateArray());

        Assert.Equal("hosted", entry.GetProperty("runnerMode").GetString());
        Assert.Equal("ubuntu:22.04", entry.GetProperty("containerImage").GetString());
        Assert.Equal("[\"ubuntu-22.04\"]", entry.GetProperty("runsOnJson").GetString());
    }

    [Theory]
    [InlineData("arm64-sbsa", "linux-arm64-sbsa")]
    [InlineData("jetson-l4t", "linux-jetson-l4t")]
    [InlineData("non-ubuntu", "non-ubuntu-linux")]
    public void FutureLinuxKeySetsFailInsteadOfDispatchingUnsupportedPackages(string keySet, string targetName)
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeKeySet.ps1");
        (int exitCode, string output) = RunPowerShellAllowFailure(
            script,
            "-Platform", "linux",
            "-RuntimeKeySet", keySet,
            "-RunnerMode", "self-hosted",
            "-OutputFormat", "json");

        Assert.NotEqual(0, exitCode);
        Assert.Contains("future package line", output, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(targetName, output, StringComparison.Ordinal);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        (int exitCode, string output) = RunPowerShellAllowFailure(scriptPath, arguments);

        Assert.True(exitCode == 0, $"PowerShell command failed with exit code {exitCode}:{Environment.NewLine}{output}");
        return output.Trim();
    }

    private static (int ExitCode, string Output) RunPowerShellAllowFailure(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = PowerShellHost.ResolveExecutable(),
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        if (OperatingSystem.IsWindows())
        {
            startInfo.ArgumentList.Add("-ExecutionPolicy");
            startInfo.ArgumentList.Add("Bypass");
        }

        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        return (process.ExitCode, output + error);
    }
}
