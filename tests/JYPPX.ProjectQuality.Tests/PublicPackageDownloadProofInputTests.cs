using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicPackageDownloadProofInputTests
{
    [Fact]
    public void PublicPackageDownloadTemplateExportsBlockedNonPublishingInputSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicPackageDownloadProofInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicPackageDownloadProofInput.ps1"), "-Strict");

        using JsonDocument templateDocument = ReadFinalReleaseJson("public-package-download-proof-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("public-package-download-proof-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-package-download-proof-required", template.GetProperty("validationState").GetString());
        Assert.Equal("JYPPX.TensorRT.CSharp.API", template.GetProperty("managedPackageId").GetString());
        Assert.StartsWith("JYPPX.TensorRT.CSharp.API.runtime.", template.GetProperty("runtimePackageId").GetString(), StringComparison.Ordinal);
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canPublishGitHubPackages").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(template.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Contains("dry-run artifacts and hashes are not public package download proof", template.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("public-package-download-proof-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("public-package-download-proof-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-package-download-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("publicPackageDownloadProofReady").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());

        string[] validationItemIds = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("source-url-public-not-local", validationItemIds);
        Assert.Contains("downloaded-managed-path-public-download", validationItemIds);
        Assert.Contains("downloaded-managed-sha256-format", validationItemIds);
        Assert.Contains("downloaded-managed-hash-match", validationItemIds);
        Assert.Contains("dry-run-sha-not-substituted", validationItemIds);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "public-package-download-proof-input.template.md"));
        Assert.Contains("Public Package Download Proof Input Template", markdown, StringComparison.Ordinal);
        Assert.Contains("Forbidden Substitutes", markdown, StringComparison.Ordinal);
        Assert.Contains("GitHub Actions package-managed-dry-run artifact", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicPackageDownloadValidatorRejectsLocalFeedDirectNupkgAndDryRunSubstitution()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicPackageDownloadProofInputTemplate.ps1"));

        using JsonDocument templateDocument = ReadFinalReleaseJson("public-package-download-proof-input.template.json");
        Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);

        values["publicPackageSourceUrl"] = Path.Combine(RepositoryPaths.Root, "artifacts", "package-managed-dry-run");
        values["publicPackageSourceKind"] = "nuget.org";
        values["downloadedManagedNupkgPath"] = templateDocument.RootElement.GetProperty("packageDryRunArtifactPath").GetString();
        values["downloadedManagedNupkgSha256"] = templateDocument.RootElement.GetProperty("packageDryRunManagedNupkgSha256").GetString();
        values["downloadedRuntimeNupkgPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "package-managed-dry-run", "runtime.nupkg");
        values["downloadedRuntimeNupkgSha256"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        values["managedPackageVersion"] = "4.0.0";
        values["runtimePackageVersion"] = "4.0.0";
        values["downloadCommand"] = "dotnet restore --source ./artifacts/package-managed-dry-run";
        values["downloadedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
        values["ownerName"] = "owner";

        string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "public-package-download-proof-input.misuse.json");
        File.WriteAllText(misusePath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicPackageDownloadProofInput.ps1"),
            "-InputPath",
            "artifacts/final-release/public-package-download-proof-input.misuse.json",
            "-Strict");

        using JsonDocument validationDocument = ReadFinalReleaseJson("public-package-download-proof-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-public-package-download-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("publicPackageDownloadProofReady").GetBoolean());

        AssertValidationItemFailed(validation, "source-url-public-not-local");
        AssertValidationItemFailed(validation, "downloaded-managed-path-public-download");
        AssertValidationItemFailed(validation, "downloaded-runtime-path-public-download");
        AssertValidationItemFailed(validation, "dry-run-sha-not-substituted");
    }

    [Fact]
    public void PublicPackageDownloadValidatorAcceptsHashMatchedPublicDownloadShapeWithoutPromotingRuntimeProof()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-public-download-proof-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string managedPackagePath = Path.Combine(tempRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
        string runtimePackagePath = Path.Combine(tempRoot, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg");

        try
        {
            CreateMinimalNupkg(managedPackagePath, "JYPPX.TensorRT.CSharp.API");
            CreateMinimalNupkg(runtimePackagePath, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22");

            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicPackageDownloadProofInputTemplate.ps1"));
            using JsonDocument templateDocument = ReadFinalReleaseJson("public-package-download-proof-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["managedPackageVersion"] = "4.0.0";
            values["runtimePackageVersion"] = "4.0.0";
            values["publicPackageSourceUrl"] = "https://api.nuget.org/v3/index.json";
            values["publicPackageSourceKind"] = "nuget.org";
            values["downloadedManagedNupkgPath"] = managedPackagePath;
            values["downloadedManagedNupkgSha256"] = Sha256(managedPackagePath);
            values["downloadedRuntimeNupkgPath"] = runtimePackagePath;
            values["downloadedRuntimeNupkgSha256"] = Sha256(runtimePackagePath);
            values["downloadCommand"] = "dotnet restore --source https://api.nuget.org/v3/index.json";
            values["downloadedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["ownerName"] = "owner";
            values["ownerAuthorizationState"] = "owner-reviewed-download-only";
            values["isPublishedPackageProof"] = false;

            string readyPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "public-package-download-proof-input.ready.json");
            File.WriteAllText(readyPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicPackageDownloadProofInput.ps1"),
                "-InputPath",
                "artifacts/final-release/public-package-download-proof-input.ready.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("public-package-download-proof-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("public-package-download-proof-input-ready", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(validation.GetProperty("publicPackageDownloadProofReady").GetBoolean());
            Assert.False(validation.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static void AssertValidationItemFailed(JsonElement validation, string itemId)
    {
        JsonElement item = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Single(candidate => candidate.GetProperty("id").GetString() == itemId);
        Assert.False(item.GetProperty("passed").GetBoolean());
    }

    private static Dictionary<string, object?> ToDictionary(JsonElement element)
    {
        return element.EnumerateObject().ToDictionary(
            static property => property.Name,
            static property => property.Value.ValueKind switch
            {
                JsonValueKind.True => (object?)true,
                JsonValueKind.False => false,
                JsonValueKind.Array => property.Value.EnumerateArray().Select(static item => item.GetString()).ToArray(),
                _ => property.Value.GetString()
            });
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void CreateMinimalNupkg(string path, string packageId)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        AddZipEntry(archive, "_rels/.rels", "<Relationships />");
        AddZipEntry(archive, $"{packageId}.nuspec", $"<package><metadata><id>{packageId}</id><version>4.0.0</version></metadata></package>");
        AddZipEntry(archive, "README.md", "# package");
    }

    private static void AddZipEntry(ZipArchive archive, string entryName, string content)
    {
        ZipArchiveEntry entry = archive.CreateEntry(entryName);
        using Stream stream = entry.Open();
        using StreamWriter writer = new(stream);
        writer.Write(content);
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream)).ToLowerInvariant();
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
