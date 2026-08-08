using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicationLicenseReadinessTests
{
    [Fact]
    public void StaticPolicyRecordsOwnerLicenseAndKeepsPublicationExplicit()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");
        (int exitCode, string output) = RunPowerShell(script, "-StaticOnly");

        Assert.Equal(0, exitCode);
        using JsonDocument document = JsonDocument.Parse(ExtractJson(output));
        Assert.True(document.RootElement.GetProperty("passed").GetBoolean());
        Assert.Equal("approved", document.RootElement.GetProperty("ownerDecisionState").GetString());
        Assert.Equal("expression", document.RootElement.GetProperty("selectedPackageLicenseType").GetString());
        Assert.Equal("Apache-2.0", document.RootElement.GetProperty("selectedPackageLicenseValue").GetString());
        Assert.Equal("LICENSE", document.RootElement.GetProperty("selectedSourceArchiveLicenseFileName").GetString());
        Assert.False(document.RootElement.GetProperty("performsPublish").GetBoolean());

        string pushScript = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Push-NuGetPackages.ps1"));
        int licenseGate = pushScript.IndexOf("Test-PublicationLicenseReadiness.ps1", StringComparison.Ordinal);
        int pushLoop = pushScript.IndexOf("while (-not $pushed", StringComparison.Ordinal);
        Assert.True(licenseGate >= 0 && pushLoop > licenseGate);

        foreach (string workflowName in new[]
        {
            "package-managed.yml",
            "package-source.yml",
            "runtime-windows.yml",
            "runtime-linux.yml",
            "release-bundle.yml",
        })
        {
            string workflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", workflowName));
            int workflowGate = workflow.IndexOf("Test-PublicationLicenseReadiness.ps1", StringComparison.Ordinal);
            Assert.True(workflowGate >= 0);
            int firstCreate = workflow.IndexOf("gh release create", StringComparison.OrdinalIgnoreCase);
            int firstUpload = workflow.IndexOf("gh release upload", StringComparison.OrdinalIgnoreCase);
            int firstPublicationCommand = new[] { firstCreate, firstUpload }.Where(static index => index >= 0).DefaultIfEmpty(int.MaxValue).Min();
            Assert.True(workflowGate < firstPublicationCommand);
        }

        string sourceWorkflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "package-source.yml"));
        Assert.Matches("owner_publish_approved:[\\s\\S]*?default: false", sourceWorkflow);
        Assert.Contains("Source archive publication requires owner_publish_approved=true", sourceWorkflow, StringComparison.Ordinal);
        Assert.Contains(
            "inputs.attach_to_github_release && inputs.release_tag != '' && inputs.owner_publish_approved && github.repository_owner == 'guojin-yan'",
            sourceWorkflow,
            StringComparison.Ordinal);
        Assert.Contains("permissions:\n  contents: read", sourceWorkflow.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);

        string bundleWorkflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));
        Assert.Contains("owner_publish_approved=$OWNER_PUBLISH_APPROVED", bundleWorkflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Ensure GitHub Release Exists", bundleWorkflow, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageWithoutLicenseMetadataIsRejected()
    {
        string tempRoot = CreateTempRoot();
        try
        {
            CreatePackage(Path.Combine(tempRoot, "no-license.nupkg"), "JYPPX.NoLicense", null, null, false);
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");
            (int exitCode, string output) = RunPowerShell(script, "-ArtifactPath", tempRoot);

            Assert.NotEqual(0, exitCode);
            Assert.Contains("has no nuspec license metadata", output, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Owner must select a license", output, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public void ArtifactsWithADifferentLicenseThanOwnerSelectionAreRejected()
    {
        string tempRoot = CreateTempRoot();
        try
        {
            CreatePackage(Path.Combine(tempRoot, "expression.nupkg"), "JYPPX.Expression", "expression", "MIT", false);
            CreateSourceArchive(Path.Combine(tempRoot, "TensorRtSharp4.0-source-4.0.0.zip"), includeLicense: true);
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");
            (int exitCode, string output) = RunPowerShell(script, "-ArtifactPath", tempRoot);

            Assert.NotEqual(0, exitCode);
            Assert.Contains(
                ReadFailures(output),
                static failure => failure.Contains("does not match the Owner-selected license", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public void MissingPackageLicenseFileAndUnlicensedSourceArchiveAreRejected()
    {
        string packageRoot = CreateTempRoot();
        string sourceRoot = CreateTempRoot();
        try
        {
            CreatePackage(Path.Combine(packageRoot, "missing-license-file.nupkg"), "JYPPX.MissingFile", "file", "LICENSE.txt", false);
            CreateSourceArchive(Path.Combine(sourceRoot, "TensorRtSharp4.0-source-4.0.0.zip"), includeLicense: false);
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");

            (int packageExitCode, string packageOutput) = RunPowerShell(script, "-ArtifactPath", packageRoot);
            Assert.NotEqual(0, packageExitCode);
            Assert.Contains(
                ReadFailures(packageOutput),
                static failure => failure.Contains("license file 'LICENSE.txt' is missing or empty", StringComparison.OrdinalIgnoreCase));

            (int sourceExitCode, string sourceOutput) = RunPowerShell(script, "-ArtifactPath", sourceRoot);
            Assert.NotEqual(0, sourceExitCode);
            Assert.Contains(
                ReadFailures(sourceOutput),
                static failure => failure.Contains("has no non-empty root license file", StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            Directory.Delete(packageRoot, recursive: true);
            Directory.Delete(sourceRoot, recursive: true);
        }
    }

    [Fact]
    public void MatchingExpressionAndSourceLicensesAreAcceptedWithoutPublishing()
    {
        string tempRoot = CreateTempRoot();
        string repositoryRoot = CreateApprovedPolicyRepository("expression", "MIT", "LICENSE");
        try
        {
            CreatePackage(Path.Combine(tempRoot, "expression.nupkg"), "JYPPX.Expression", "expression", "MIT", false);
            CreateSourceArchive(Path.Combine(tempRoot, "TensorRtSharp4.0-source-4.0.0.zip"), includeLicense: true);
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");
            (int exitCode, string output) = RunPowerShell(
                script,
                "-RepositoryRoot", repositoryRoot,
                "-ArtifactPath", tempRoot);

            Assert.Equal(0, exitCode);
            using JsonDocument document = JsonDocument.Parse(ExtractJson(output));
            JsonElement root = document.RootElement;
            Assert.True(root.GetProperty("passed").GetBoolean());
            Assert.Equal(2, root.GetProperty("inspectedArtifactCount").GetInt32());
            Assert.False(root.GetProperty("performsPublish").GetBoolean());
            Assert.Contains(
                root.GetProperty("inspectedArtifacts").EnumerateArray(),
                static item => item.GetProperty("licenseType").GetString() == "expression" &&
                    item.GetProperty("licenseValue").GetString() == "MIT");
            Assert.Contains(
                root.GetProperty("inspectedArtifacts").EnumerateArray(),
                static item => item.GetProperty("artifactType").GetString() == "source-archive" &&
                    item.GetProperty("licenseEntry").GetString()!.EndsWith("/LICENSE", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
            Directory.Delete(repositoryRoot, recursive: true);
        }
    }

    [Fact]
    public void MatchingFileAndSourceLicensesAreAcceptedWithoutPublishing()
    {
        string tempRoot = CreateTempRoot();
        string repositoryRoot = CreateApprovedPolicyRepository("file", "LICENSE.txt", "LICENSE");
        try
        {
            CreatePackage(Path.Combine(tempRoot, "file.nupkg"), "JYPPX.File", "file", "LICENSE.txt", true);
            CreateSourceArchive(Path.Combine(tempRoot, "TensorRtSharp4.0-source-4.0.0.zip"), includeLicense: true);
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");
            (int exitCode, string output) = RunPowerShell(
                script,
                "-RepositoryRoot", repositoryRoot,
                "-ArtifactPath", tempRoot);

            Assert.Equal(0, exitCode);
            using JsonDocument document = JsonDocument.Parse(ExtractJson(output));
            JsonElement[] artifacts = document.RootElement.GetProperty("inspectedArtifacts").EnumerateArray().ToArray();
            Assert.Contains(
                artifacts,
                static item => item.GetProperty("artifactType").GetString() == "nupkg" &&
                    item.GetProperty("licenseEntry").GetString() == "LICENSE.txt");
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
            Directory.Delete(repositoryRoot, recursive: true);
        }
    }

    [Fact]
    public void UnsafePackageAndSourceArchiveLicensePathsAreRejected()
    {
        string tempRoot = CreateTempRoot();
        string repositoryRoot = CreateApprovedPolicyRepository("file", "LICENSE.txt", "LICENSE");
        try
        {
            CreatePackage(Path.Combine(tempRoot, "unsafe.nupkg"), "JYPPX.Unsafe", "file", "../LICENSE.txt", true);
            CreateSourceArchive(
                Path.Combine(tempRoot, "TensorRtSharp4.0-source-4.0.0.zip"),
                includeLicense: false,
                licenseEntryName: "../LICENSE");
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicationLicenseReadiness.ps1");
            (int exitCode, string output) = RunPowerShell(
                script,
                "-RepositoryRoot", repositoryRoot,
                "-ArtifactPath", tempRoot);

            Assert.NotEqual(0, exitCode);
            string[] failures = ReadFailures(output);
            Assert.Contains(failures, static failure => failure.Contains("unsafe license file path", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(failures, static failure => failure.Contains("has no non-empty root license file", StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
            Directory.Delete(repositoryRoot, recursive: true);
        }
    }

    private static string CreateTempRoot()
    {
        string path = Path.Combine(Path.GetTempPath(), "jyppx-publication-license-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static string CreateApprovedPolicyRepository(
        string packageLicenseType,
        string packageLicenseValue,
        string sourceLicenseFileName)
    {
        string root = CreateTempRoot();
        Directory.CreateDirectory(Path.Combine(root, "pack"));
        Directory.CreateDirectory(Path.Combine(root, "eng"));
        Directory.CreateDirectory(Path.Combine(root, ".github", "workflows"));
        File.WriteAllText(
            Path.Combine(root, "pack", "publication-license-policy.json"),
            $$"""
            {
              "schemaVersion": 1,
              "policyId": "declared-license-required-before-publication",
              "ownerDecisionState": "approved",
              "publicationRequiresDeclaredLicense": true,
              "dryRunMayProceedWithoutDeclaredLicense": true,
              "acceptedPackageLicenseTypes": [ "expression", "file" ],
              "disallowedLicenseValues": [ "", "NOASSERTION", "NONE", "UNLICENSED", "TBD", "TODO" ],
              "sourceArchiveLicenseFileNames": [ "LICENSE", "LICENSE.txt" ],
              "selectedPackageLicense": {
                "type": "{{packageLicenseType}}",
                "value": "{{packageLicenseValue}}"
              },
              "selectedSourceArchiveLicenseFileName": "{{sourceLicenseFileName}}"
            }
            """);
        File.WriteAllText(
            Path.Combine(root, "eng", "Push-NuGetPackages.ps1"),
            "Test-PublicationLicenseReadiness.ps1");
        foreach (string workflowName in new[]
        {
            "package-managed.yml",
            "package-source.yml",
            "runtime-windows.yml",
            "runtime-linux.yml",
            "release-bundle.yml",
        })
        {
            File.WriteAllText(
                Path.Combine(root, ".github", "workflows", workflowName),
                "Test-PublicationLicenseReadiness.ps1");
        }

        return root;
    }

    private static void CreatePackage(
        string path,
        string packageId,
        string? licenseType,
        string? licenseValue,
        bool includeLicenseFile)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        string license = licenseType is null
            ? string.Empty
            : $"<license type=\"{licenseType}\">{licenseValue}</license>";
        AddEntry(
            archive,
            $"{packageId}.nuspec",
            $"<package><metadata><id>{packageId}</id><version>4.0.0</version>{license}</metadata></package>");
        if (includeLicenseFile)
        {
            AddEntry(archive, licenseValue!, "fixture license text");
        }
    }

    private static void CreateSourceArchive(
        string path,
        bool includeLicense,
        string? licenseEntryName = null)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        AddEntry(archive, "TensorRtSharp4.0-4.0.0/README.md", "fixture readme");
        if (includeLicense)
        {
            AddEntry(archive, "TensorRtSharp4.0-4.0.0/LICENSE", "fixture license text");
        }
        else if (licenseEntryName is not null)
        {
            AddEntry(archive, licenseEntryName, "fixture license text");
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
            FileName = PowerShellHost.ResolveExecutable(),
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

    private static string[] ReadFailures(string output)
    {
        using JsonDocument document = JsonDocument.Parse(ExtractJson(output));
        return document.RootElement
            .GetProperty("failures")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
    }
}
