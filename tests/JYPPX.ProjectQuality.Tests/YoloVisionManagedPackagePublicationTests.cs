using System.Diagnostics;
using System.IO.Compression;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionManagedPackagePublicationTests
{
    [Fact]
    public void PackageManagedWorkflowPacksAndValidatesTheExactManagedBundle()
    {
        string workflow = ReadSource(".github", "workflows", "package-managed.yml");
        string normalized = workflow.Replace("\r\n", "\n", StringComparison.Ordinal);

        Assert.Contains("permissions:\n  contents: read", normalized, StringComparison.Ordinal);
        Assert.Contains("owner_publish_approved", workflow, StringComparison.Ordinal);
        Assert.Contains("Package or Release publication requires owner_publish_approved=true", workflow, StringComparison.Ordinal);
        Assert.Contains("Pack managed package", workflow, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Pack YoloVision managed extension", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Pack Classification managed extension", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Test-YoloVisionPackageSurface.ps1", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Test-YoloVisionManagedPackageDryRun.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("FullyQualifiedName~YoloVisionManagedPackagePublicationTests", workflow, StringComparison.Ordinal);
        Assert.Contains("-RequireExactPackageSet", workflow, StringComparison.Ordinal);
        Assert.Contains("Expected exactly one core managed", workflow, StringComparison.Ordinal);
        Assert.Contains("attach-github-release:", workflow, StringComparison.Ordinal);
        Assert.Contains("inputs.attach_to_github_release && inputs.release_tag != '' && inputs.owner_publish_approved && github.repository_owner == 'guojin-yan'", workflow, StringComparison.Ordinal);
        Assert.Contains("Failed to create GitHub Release", workflow, StringComparison.Ordinal);
        Assert.Contains("Failed to upload managed package", workflow, StringComparison.Ordinal);
        Assert.Contains("inputs.publish_to_github_packages && inputs.owner_publish_approved && github.repository_owner == 'guojin-yan'", workflow, StringComparison.Ordinal);
        Assert.Contains("inputs.publish_to_nuget && inputs.owner_publish_approved && github.repository_owner == 'guojin-yan'", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidationAccountCanRunReadOnlyPackageDryRun()
    {
        string workflow = ReadSource(".github", "workflows", "release-quality-gate.yml");

        Assert.Contains("if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_package_managed_dry_run }}", workflow, StringComparison.Ordinal);
        Assert.Contains("owner_publish_approved: false", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_nuget: false", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_github_packages: false", workflow, StringComparison.Ordinal);
        Assert.Contains("attach_to_github_release: false", workflow, StringComparison.Ordinal);
        Assert.Contains("package-managed-dry-run:\n", workflow.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);
        Assert.Contains("is still downgraded by package-managed.yml to contents: read", workflow, StringComparison.Ordinal);
        Assert.Contains("contents: write", workflow, StringComparison.Ordinal);
        Assert.Contains("packages: write", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("github.repository_owner == 'guojin-yan' && github.event_name == 'workflow_dispatch' && inputs.run_package_managed_dry_run", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseBundleDefaultsToNoPublicationSideEffects()
    {
        string workflow = ReadSource(".github", "workflows", "release-bundle.yml");
        string wrapper = ReadSource("eng", "Invoke-RemoteReleaseBundle.ps1");

        Assert.Matches("owner_publish_approved:[\\s\\S]*?default: false", workflow);
        Assert.Matches("run_docs_release:[\\s\\S]*?default: false", workflow);
        Assert.Matches("publish_managed_to_github_packages:[\\s\\S]*?default: false", workflow);
        Assert.Matches("attach_runtime_to_github_release:[\\s\\S]*?default: false", workflow);
        string dispatchInputs = workflow[
            (workflow.IndexOf("    inputs:", StringComparison.Ordinal) + "    inputs:".Length)..
            workflow.IndexOf("\npermissions:", StringComparison.Ordinal)];
        int dispatchInputCount = System.Text.RegularExpressions.Regex.Matches(
            dispatchInputs,
            "(?m)^      [a-z0-9_]+:$").Count;
        Assert.InRange(dispatchInputCount, 1, 25);
        Assert.DoesNotContain("      windows_cuda_cudnn_package_version:\n", workflow.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);
        Assert.DoesNotContain("      windows_tensorrt_package_version:\n", workflow.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);
        Assert.Contains("Release publication side effects require owner_publish_approved=true", workflow, StringComparison.Ordinal);
        Assert.Contains("owner_publish_approved=$OWNER_PUBLISH_APPROVED", workflow, StringComparison.Ordinal);
        Assert.Contains("[object]$OwnerPublishApproved = $false", wrapper, StringComparison.Ordinal);
        Assert.Contains("[object]$RunDocsRelease = $false", wrapper, StringComparison.Ordinal);
        Assert.Contains("[object]$PublishManagedToGitHubPackages = $false", wrapper, StringComparison.Ordinal);
        Assert.Contains("[object]$AttachRuntimeToGitHubRelease = $false", wrapper, StringComparison.Ordinal);
        Assert.Contains("require -OwnerPublishApproved true", wrapper, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ManagedDryRunAndThreePackageHandoffKeepProofBoundariesExplicit()
    {
        string dryRun = ReadSource("eng", "Test-YoloVisionManagedPackageDryRun.ps1");
        string handoff = ReadSource("eng", "Export-YoloVisionPackagePublicationHandoff.ps1");
        string project = ReadSource("samples", "YoloVision.ManagedPackageConsumer", "YoloVision.ManagedPackageConsumer.csproj.template");
        string program = ReadSource("samples", "YoloVision.ManagedPackageConsumer", "Program.cs");

        Assert.Equal(2, CountOccurrences(project, "<PackageReference"));
        Assert.DoesNotContain("ProjectReference", project, StringComparison.Ordinal);
        Assert.Contains("Process.GetCurrentProcess().Modules", program, StringComparison.Ordinal);
        Assert.Contains("jyppxtrtbridge", program, StringComparison.Ordinal);
        Assert.Contains("NativeRuntimeLoaded={nativeRuntimeLoaded}", program, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"yolovision-managed-package-dry-run\"", dryRun, StringComparison.Ordinal);
        Assert.Contains("OutputRoot must be outside the repository", dryRun, StringComparison.Ordinal);
        Assert.Contains("-PackagePath $PackageDirectory", dryRun, StringComparison.Ordinal);
        Assert.Contains("restoredProjectLibraryCount = $projectLibraryCount", dryRun, StringComparison.Ordinal);
        Assert.Contains("packageSourceCommitsAligned = $true", dryRun, StringComparison.Ordinal);
        Assert.Contains("isTensorRtRuntimeProof = $false", dryRun, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"yolovision-package-publication-handoff\"", handoff, StringComparison.Ordinal);
        Assert.Contains("ready-local-three-package-handoff", handoff, StringComparison.Ordinal);
        Assert.Contains("packageSourceCommitsAligned = $true", handoff, StringComparison.Ordinal);
        Assert.Contains("postPublishCleanConsumerRequired = $true", handoff, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", handoff, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateInventoryRequiresYoloVisionManagedExtension()
    {
        string inventory = ReadSource("eng", "Export-ReleaseCandidatePackageInventory.ps1");

        Assert.Contains("managed-extension", inventory, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API.YoloVision", inventory, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API.Classification", inventory, StringComparison.Ordinal);
        Assert.Contains("managedExtensionPackageReady", inventory, StringComparison.Ordinal);
        Assert.Contains("packageSourceCommitsAligned", inventory, StringComparison.Ordinal);
        Assert.Contains("packageVersionsAligned", inventory, StringComparison.Ordinal);
        Assert.Contains("$candidateAllowedPackages.Count -eq $allowedPackages.Count", inventory, StringComparison.Ordinal);
        Assert.Contains("$requiredManagedExtensionPackageIds -contains $_.packageId", inventory, StringComparison.Ordinal);
        Assert.Contains("explicit managed extensions", inventory, StringComparison.Ordinal);
    }

    [Fact]
    public void ExactPackageSetPolicyRejectsMissingUnexpectedAndWrongVersionPackages()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-managed-allowlist-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        try
        {
            string managed = Path.Combine(tempRoot, "managed.nupkg");
            string bridge = Path.Combine(tempRoot, "bridge.nupkg");
            CreatePackage(managed, "JYPPX.TensorRT.CSharp.API", "4.0.0");
            CreatePackage(bridge, "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge", "4.0.0", "runtimes/win-x64/native/jyppxtrtbridge.dll");

            string policy = Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalVendorRuntimePackagePolicy.ps1");
            string[] common =
            [
                "-PackagePath", tempRoot,
                "-ExpectedPackageId", "JYPPX.TensorRT.CSharp.API,JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge",
                "-ExpectedPackageVersion", "4.0.0",
                "-RequireExactPackageSet",
            ];
            (int validExit, string validOutput) = RunPowerShell(policy, common);
            Assert.Equal(0, validExit);
            Assert.Contains("\"requireExactPackageSet\":  true", validOutput, StringComparison.Ordinal);

            File.Delete(bridge);
            (int missingExit, string missingOutput) = RunPowerShell(policy, common);
            Assert.NotEqual(0, missingExit);
            Assert.Contains("Expected package set contains 2 package(s), but inspected 1", missingOutput, StringComparison.Ordinal);

            CreatePackage(bridge, "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge", "4.0.1", "runtimes/win-x64/native/jyppxtrtbridge.dll");
            (int wrongVersionExit, string wrongVersionOutput) = RunPowerShell(policy, common);
            Assert.NotEqual(0, wrongVersionExit);
            Assert.Contains("4.0.1", wrongVersionOutput, StringComparison.Ordinal);
            Assert.Contains("4.0.0", wrongVersionOutput, StringComparison.Ordinal);

            File.Delete(bridge);
            CreatePackage(bridge, "Forbidden.Package", "4.0.0");
            (int unexpectedExit, string unexpectedOutput) = RunPowerShell(policy, common);
            Assert.NotEqual(0, unexpectedExit);
            Assert.Contains("outside the expected publication allowlist", unexpectedOutput, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }

    private static int CountOccurrences(string text, string value)
    {
        int count = 0;
        int index = 0;
        while ((index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }
        return count;
    }

    private static void CreatePackage(string path, string packageId, string version, params string[] nativeEntries)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        ZipArchiveEntry entry = archive.CreateEntry($"{packageId}.nuspec");
        using (StreamWriter writer = new(entry.Open()))
        {
            writer.Write($"<package><metadata><id>{packageId}</id><version>{version}</version></metadata></package>");
        }
        foreach (string nativeEntry in nativeEntries)
        {
            ZipArchiveEntry native = archive.CreateEntry(nativeEntry);
            using (StreamWriter nativeWriter = new(native.Open()))
            {
                nativeWriter.Write("bridge");
            }
        }
    }

    private static (int ExitCode, string Output) RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = OperatingSystem.IsWindows() ? "powershell" : "pwsh",
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
}
