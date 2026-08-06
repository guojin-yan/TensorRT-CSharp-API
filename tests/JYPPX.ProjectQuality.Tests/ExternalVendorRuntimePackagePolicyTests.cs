using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ExternalVendorRuntimePackagePolicyTests
{
    [Fact]
    public void PublicationPolicyAllowsOnlyManagedBridgeAndTrackedSourceArtifacts()
    {
        string policyPath = Path.Combine(RepositoryPaths.Root, "pack", "external-vendor-runtime-policy.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(policyPath));
        JsonElement root = document.RootElement;

        Assert.Equal("external-vendor-runtime-not-redistributed", root.GetProperty("policyId").GetString());
        string?[] allowedKinds = root.GetProperty("allowedPackageKinds").EnumerateArray().Select(static value => value.GetString()).ToArray();
        Assert.Collection(
            allowedKinds,
            static value => Assert.Equal("managed", value),
            static value => Assert.Equal("bridge", value));
        Assert.Equal("guojin-yan", root.GetProperty("formalReleaseRepositoryOwner").GetString());
        Assert.Equal("grape-yan", root.GetProperty("validationOnlyRepositoryOwner").GetString());
        Assert.Equal(
            new[] { "JYPPX.TensorRT.CSharp.API" },
            root.GetProperty("managedPackageIds").EnumerateArray().Select(static value => value.GetString()).ToArray());
        Assert.Equal(
            new[] { "JYPPX.TensorRT.CSharp.API.YoloVision", "JYPPX.TensorRT.CSharp.API.Classification" },
            root.GetProperty("nonPublicSamplePackageIds").EnumerateArray().Select(static value => value.GetString()).ToArray());
        Assert.Contains(
            root.GetProperty("forbiddenPublicClaims").EnumerateArray(),
            static value => value.GetString() == "GitHub 全依赖包");
        Assert.True(root.GetProperty("sourceArchivePolicy").GetProperty("mustUseGitTrackedFiles").GetBoolean());
        Assert.True(root.GetProperty("sourceArchivePolicy").GetProperty("mustExcludeThirdPartyBinaries").GetBoolean());

        string runtimeProps = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "pack", "runtime", "Directory.Build.props"));
        string splitProps = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "pack", "runtime-split", "Directory.Build.props"));
        Assert.Contains("<IsPackable>false</IsPackable>", runtimeProps, StringComparison.Ordinal);
        Assert.Contains("<IsPackable>false</IsPackable>", splitProps, StringComparison.Ordinal);
        Assert.Contains("Requires user-installed matching CUDA, cuDNN, and TensorRT", splitProps, StringComparison.Ordinal);
        Assert.DoesNotContain("Retired split runtime project", splitProps, StringComparison.Ordinal);

        string[] splitProjects = Directory.GetFiles(Path.Combine(RepositoryPaths.Root, "pack", "runtime-split"), "*.csproj", SearchOption.AllDirectories);
        foreach (string projectPath in splitProjects)
        {
            string xml = XDocument.Load(projectPath).ToString(SaveOptions.DisableFormatting);
            bool isBridge = Path.GetFileNameWithoutExtension(projectPath).EndsWith(".Bridge", StringComparison.Ordinal);
            if (isBridge)
            {
                Assert.Contains("<JYPPXPackageKind>bridge</JYPPXPackageKind>", xml, StringComparison.Ordinal);
                Assert.Contains("<IsPackable>true</IsPackable>", xml, StringComparison.Ordinal);
            }
            else
            {
                Assert.DoesNotContain("<IsPackable>true</IsPackable>", xml, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void BuildAndDependencyDiscoveryDoNotProbeRepositoryVendorDrops()
    {
        string[] files =
        [
            "CMakeLists.txt",
            Path.Combine("eng", "Get-Dependencies.ps1"),
            Path.Combine("eng", "get-dependencies.sh"),
            Path.Combine("eng", "Resolve-RuntimeRoots.ps1"),
            Path.Combine("eng", "Sync-LocalRuntimeRoots.ps1"),
            Path.Combine("eng", "Export-InterfaceCoverageMatrix.ps1"),
            Path.Combine("pack", "runtime", "runtime-packages.local.example.json"),
            Path.Combine("third_party", "README.md"),
        ];

        foreach (string relativePath in files)
        {
            string content = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath))
                .Replace('\\', '/');
            Assert.DoesNotContain("third_party/nvidia", content, StringComparison.OrdinalIgnoreCase);
        }

        string roots = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeRoots.ps1"));
        Assert.Contains("$env:JYPPX_TENSORRT_ROOT", roots, StringComparison.Ordinal);
        Assert.Contains("$env:JYPPX_CUDA_ROOT", roots, StringComparison.Ordinal);
        Assert.Contains("$env:JYPPX_CUDNN_ROOT", roots, StringComparison.Ordinal);
    }

    [Fact]
    public void StaticPolicyGateAndCleanupPlanPassWithoutRemoteSideEffects()
    {
        string gate = Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalVendorRuntimePackagePolicy.ps1");
        (int gateExitCode, string gateOutput) = RunPowerShell(gate, "-StaticOnly");
        Assert.Equal(0, gateExitCode);
        using JsonDocument gateDocument = JsonDocument.Parse(ExtractJson(gateOutput));
        Assert.True(gateDocument.RootElement.GetProperty("passed").GetBoolean());
        Assert.Equal(0, gateDocument.RootElement.GetProperty("failureCount").GetInt32());

        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-retired-package-plan-" + Guid.NewGuid().ToString("N"));
        try
        {
            string exporter = Path.Combine(RepositoryPaths.Root, "eng", "Export-RetiredVendorPackageCleanupPlan.ps1");
            (int exportExitCode, string exportOutput) = RunPowerShell(exporter, "-OutputDirectory", tempRoot);
            Assert.Equal(0, exportExitCode);
            using JsonDocument planDocument = JsonDocument.Parse(ExtractJson(exportOutput));
            JsonElement plan = planDocument.RootElement;
            Assert.True(plan.GetProperty("candidateCount").GetInt32() > 10);
            Assert.False(plan.GetProperty("remoteInventoryComplete").GetBoolean());
            Assert.False(plan.GetProperty("deleteExecuted").GetBoolean());
            Assert.False(plan.GetProperty("performsRemoteQuery").GetBoolean());
            Assert.False(plan.GetProperty("performsDelete").GetBoolean());
            Assert.Contains(plan.GetProperty("candidates").EnumerateArray(), static candidate => candidate.GetProperty("role").GetString() == "full-runtime");
            Assert.Contains(plan.GetProperty("candidates").EnumerateArray(), static candidate => candidate.GetProperty("role").GetString() == "cuda-cudnn");
            Assert.Contains(plan.GetProperty("candidates").EnumerateArray(), static candidate => candidate.GetProperty("role").GetString() == "tensorrt");
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void PackageContentGateAcceptsBridgeAndRejectsVendorBinary()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-package-policy-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        try
        {
            string bridgePackage = Path.Combine(tempRoot, "bridge.nupkg");
            CreatePackage(
                bridgePackage,
                "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge",
                "runtimes/win-x64/native/jyppxtrtbridge.dll");

            string gate = Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalVendorRuntimePackagePolicy.ps1");
            (int bridgeExitCode, string bridgeOutput) = RunPowerShell(gate, "-PackagePath", bridgePackage);
            Assert.Equal(0, bridgeExitCode);
            using (JsonDocument bridgeDocument = JsonDocument.Parse(ExtractJson(bridgeOutput)))
            {
                Assert.True(bridgeDocument.RootElement.GetProperty("passed").GetBoolean());
            }

            string invalidYoloVisionPackage = Path.Combine(tempRoot, "yolovision-native.nupkg");
            CreatePackage(
                invalidYoloVisionPackage,
                "JYPPX.TensorRT.CSharp.API.YoloVision",
                "runtimes/win-x64/native/jyppxtrtbridge.dll");
            (int invalidYoloVisionExitCode, string invalidYoloVisionOutput) = RunPowerShell(gate, "-PackagePath", invalidYoloVisionPackage);
            Assert.NotEqual(0, invalidYoloVisionExitCode);
            Assert.Contains("Package id is not allowed by policy", invalidYoloVisionOutput, StringComparison.OrdinalIgnoreCase);

            string vendorPackage = Path.Combine(tempRoot, "vendor.nupkg");
            CreatePackage(
                vendorPackage,
                "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.CudaCudnn",
                "runtimes/win-x64/native/cudart64_12.dll",
                "runtimes/linux-x64/native/libnvinfer.so.11");

            (int vendorExitCode, string vendorOutput) = RunPowerShell(gate, "-PackagePath", vendorPackage);
            Assert.NotEqual(0, vendorExitCode);
            Assert.Contains("forbidden NVIDIA runtime binary", vendorOutput, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Package id is not allowed", vendorOutput, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public void WorkflowsAreBridgeOnlyAndValidationAccountCannotPublish()
    {
        string[] workflowNames =
        [
            "ci-validation.yml",
            "package-managed.yml",
            "package-source.yml",
            "runtime-windows.yml",
            "runtime-linux.yml",
            "release-bundle.yml",
            "release-quality-gate.yml",
        ];

        foreach (string workflowName in workflowNames)
        {
            string workflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", workflowName));
            Assert.Contains("Test-ExternalVendorRuntimePackagePolicy.ps1", workflow, StringComparison.Ordinal);
        }

        string windows = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-windows.yml"));
        string linux = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml"));
        string bundle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));
        string validation = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "ci-validation.yml"));

        Assert.DoesNotContain("default: all", windows, StringComparison.Ordinal);
        Assert.DoesNotContain("default: all", linux, StringComparison.Ordinal);
        Assert.Contains("default: bridge", windows, StringComparison.Ordinal);
        Assert.Contains("default: bridge", linux, StringComparison.Ordinal);
        Assert.Contains("dispatch_workflow \"package-source\"", bundle, StringComparison.Ordinal);
        Assert.Contains("grape-yan repository is validation-only", bundle, StringComparison.Ordinal);
        Assert.Contains("permissions:\n  contents: read", validation.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);
        Assert.DoesNotContain("Push-NuGetPackages", validation, StringComparison.Ordinal);
        Assert.DoesNotContain("gh release", validation, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PublicReleaseConsumerUsesVerifiedManagedAndBridgeAssetsOnly()
    {
        string consumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-PublicReleaseBridgePackageConsumer.ps1"));
        string routePlan = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Export-PackageConsumerDualRouteProofPlan.ps1"));

        foreach (string marker in new[]
        {
            "browser_download_url",
            "^sha256:[0-9a-fA-F]{64}$",
            "Get-FileHash",
            "Assert-PackageIdentity",
            "repositoryCommit",
            "packageSourceCommitAligned",
            "-AllowCrossCommitPair is diagnostic-only",
            "Test-ExternalVendorRuntimePackagePolicy.ps1",
            "Test-BridgePackageRuntimeConsumer.ps1",
            "restoreUsesDownloadedAssetStaging = $true",
            "stagingIsLocallyBuiltPackageFeed = $false",
            "directNupkgReferenceUsed = $false",
            "vendorRuntimeBundled = $false",
            "currentHeadBindingVerified = $false",
            "isPackageConsumerRuntimeProof = $false",
            "isPostPublishProof = $false",
            "runtime-consumer-invocation.stdout.log",
            "runtime-consumer-invocation.stderr.log",
            "-SkipInstalledVendorAssetHashing",
        })
        {
            Assert.Contains(marker, consumer, StringComparison.Ordinal);
        }

        Assert.Contains("github-release-managed-plus-bridge-assets", routePlan, StringComparison.Ordinal);
        Assert.Contains("nuget-managed-plus-bridge-packages", routePlan, StringComparison.Ordinal);
        Assert.Contains("NVIDIA vendor runtime packages are not a supported route", routePlan, StringComparison.Ordinal);
        Assert.Contains("same source commit", routePlan, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("github-full-dependency-package", routePlan, StringComparison.Ordinal);
        Assert.DoesNotContain("runtime dependency bundle", routePlan, StringComparison.OrdinalIgnoreCase);

        string runtimeConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-BridgePackageRuntimeConsumer.ps1"));
        Assert.Contains("[switch]$SkipInstalledVendorAssetHashing", runtimeConsumer, StringComparison.Ordinal);
        Assert.Contains("installedVendorAssetHashingSkipped", runtimeConsumer, StringComparison.Ordinal);
        Assert.Contains("installedVendorAssetInventorySkipped", runtimeConsumer, StringComparison.Ordinal);
        Assert.Contains("nativeAssetHashesComplete", runtimeConsumer, StringComparison.Ordinal);
        Assert.Contains("canPromoteCompatibleHostRuntimeProof = $runtimeSmokePassed -and -not $SkipInstalledVendorAssetHashing.IsPresent", runtimeConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void RemoteCleanupInventoryIsReadOnlyAndRequiresOwnerReview()
    {
        string script = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Export-RetiredVendorPackageRemoteInventory.ps1"));

        Assert.Contains("users/$Owner/packages?package_type=nuget", script, StringComparison.Ordinal);
        Assert.Contains("/versions?per_page=100", script, StringComparison.Ordinal);
        Assert.Contains("repos/$formalRepository/releases?per_page=100", script, StringComparison.Ordinal);
        Assert.Contains("remoteInventoryComplete = $true", script, StringComparison.Ordinal);
        Assert.Contains("ownerReviewRequired = $true", script, StringComparison.Ordinal);
        Assert.Contains("reviewFingerprint = $reviewFingerprint", script, StringComparison.Ordinal);
        Assert.Contains("reviewItemCount = $reviewLines.Count", script, StringComparison.Ordinal);
        Assert.Contains("performsRemoteQuery = $true", script, StringComparison.Ordinal);
        Assert.Contains("performsDelete = $false", script, StringComparison.Ordinal);
        Assert.Contains("deleteExecuted = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("--method DELETE", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Remove-Item", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CleanupExecutorDefaultsToLivePreflightAndRequiresThreeDeletionGates()
    {
        string script = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-RetiredVendorPackageCleanup.ps1"));

        Assert.Contains("SupportsShouldProcess = $true", script, StringComparison.Ordinal);
        Assert.Contains("ConfirmImpact = \"High\"", script, StringComparison.Ordinal);
        Assert.Contains("if (-not $ExecuteDeletion.IsPresent)", script, StringComparison.Ordinal);
        Assert.Contains("-ExpectedReviewFingerprint is required with -ExecuteDeletion", script, StringComparison.Ordinal);
        Assert.Contains("$ExpectedReviewFingerprint -ne $computedFingerprint", script, StringComparison.Ordinal);
        Assert.Contains("$PSCmdlet.ShouldProcess", script, StringComparison.Ordinal);
        Assert.Contains("livePreflightPassed", script, StringComparison.Ordinal);
        Assert.Contains("Unreviewed retired package exists remotely", script, StringComparison.Ordinal);
        Assert.Contains("Unreviewed package version exists", script, StringComparison.Ordinal);
        Assert.Contains("Unreviewed retired Release asset exists", script, StringComparison.Ordinal);
        Assert.Contains("$livePackageIdsByLength", script, StringComparison.Ordinal);
        Assert.Contains("Sort-Object { $_.Length } -Descending", script, StringComparison.Ordinal);
        Assert.Contains("\"--method\", \"DELETE\"", script, StringComparison.Ordinal);
        Assert.Contains("/versions/$([long]$version.id)", script, StringComparison.Ordinal);
        Assert.Contains("/releases/assets/$([long]$asset.id)", script, StringComparison.Ordinal);
        Assert.Contains("Preserved package is missing after deletion", script, StringComparison.Ordinal);
        Assert.Contains("Preserved Release asset is missing after deletion", script, StringComparison.Ordinal);
        Assert.DoesNotContain("Remove-Item", script, StringComparison.OrdinalIgnoreCase);

        int executeGuard = script.IndexOf("if (-not $ExecuteDeletion.IsPresent)", StringComparison.Ordinal);
        int shouldProcess = script.IndexOf("$PSCmdlet.ShouldProcess", StringComparison.Ordinal);
        int firstDelete = script.IndexOf("\"--method\", \"DELETE\"", StringComparison.Ordinal);
        Assert.True(executeGuard >= 0 && shouldProcess > executeGuard && firstDelete > shouldProcess);
    }

    private static void CreatePackage(string path, string packageId, params string[] nativeEntries)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        AddEntry(archive, $"{packageId}.nuspec", $"<package><metadata><id>{packageId}</id><version>4.0.0</version></metadata></package>");
        foreach (string nativeEntry in nativeEntries)
        {
            AddEntry(archive, nativeEntry, "binary");
        }
    }

    private static void AddEntry(ZipArchive archive, string name, string content)
    {
        ZipArchiveEntry entry = archive.CreateEntry(name);
        using StreamWriter writer = new(entry.Open());
        writer.Write(content);
    }

    private static (int ExitCode, string Output) RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = OperatingSystem.IsWindows() ? "powershell.exe" : "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, stdout + stderr);
    }

    private static string ExtractJson(string output)
    {
        int start = output.IndexOf('{', StringComparison.Ordinal);
        int end = output.LastIndexOf('}');
        Assert.True(start >= 0 && end >= start, $"PowerShell output did not contain JSON:{Environment.NewLine}{output}");
        return output[start..(end + 1)];
    }
}
