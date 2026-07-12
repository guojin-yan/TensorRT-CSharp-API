using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPublishAuthorizationInputTests
{
    [Fact]
    public void OwnerPublishAuthorizationTemplateExportsBlockedNonPublishingSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublishAuthorizationInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublishAuthorizationInput.ps1"), "-Strict");

        using JsonDocument templateDocument = ReadFinalReleaseJson("owner-publish-authorization-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("owner-publish-authorization-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-publish-authorization-required", template.GetProperty("validationState").GetString());
        Assert.Equal("owner-authorization-required", template.GetProperty("authorizationDecision").GetString());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("usesPublishToken").GetBoolean());
        Assert.True(template.GetProperty("requiresOwnerAuthorization").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(template.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Contains("dotnet nuget push", template.GetProperty("publishCommandTemplates").EnumerateArray().First().GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("does not execute dotnet nuget push", template.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-publish-authorization-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-publish-authorization-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-publish-authorization-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("ownerPublishAuthorizationReady").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
    }

    [Fact]
    public void OwnerPublishAuthorizationValidatorRejectsPersistedTokenAndDryRunArtifacts()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublishAuthorizationInputTemplate.ps1"));
        using JsonDocument templateDocument = ReadFinalReleaseJson("owner-publish-authorization-input.template.json");
        Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
        values["ownerName"] = "ghp_abcdefghijklmnopqrstuvwxyz123456";
        values["authorizationDecision"] = "approved-for-owner-run";
        values["authorizedRoutes"] = new[] { "nuget-small-bridge-core" };
        values["managedNupkgPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "29162977180", "package-managed-dry-run", "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
        values["runtimeNupkgPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "29162977180", "package-managed-dry-run", "runtime.nupkg");
        values["releaseNotesPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "release-notes.md");
        values["rollbackPlanPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "rollback.md");
        values["managedNupkgSha256"] = SixtyFour("a");
        values["runtimeNupkgSha256"] = SixtyFour("b");
        values["releaseNotesSha256"] = SixtyFour("c");
        values["rollbackPlanSha256"] = SixtyFour("d");
        values["ownerDecisionTimestampUtc"] = DateTimeOffset.UtcNow.ToString("O");
        values["managedPackageVersion"] = "4.0.0";
        values["runtimePackageVersion"] = "4.0.0";
        values["confirmsNoTokenPersisted"] = "true";
        values["confirmsNoDryRunArtifactSubstitution"] = "true";
        values["confirmsPackageHashesReviewed"] = "true";
        values["confirmsPublishCommandReviewed"] = "true";
        values["confirmsPublicPackageDownloadProofStillRequired"] = "true";

        string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-publish-authorization-input.misuse.json");
        File.WriteAllText(misusePath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublishAuthorizationInput.ps1"),
            "-InputPath",
            "artifacts/final-release/owner-publish-authorization-input.misuse.json");

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-publish-authorization-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-owner-publish-authorization-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("failedBlockerCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("ownerPublishAuthorizationReady").GetBoolean());
        AssertValidationItemFailed(validation, "no-token-like-secret-persisted");
        AssertValidationItemFailed(validation, "managedNupkgPath-not-dry-run-artifact");
        AssertValidationItemFailed(validation, "runtimeNupkgPath-not-dry-run-artifact");
    }

    [Fact]
    public void OwnerPublishAuthorizationValidatorAcceptsOwnerRunReadyShapeWithoutPublishingOrPostPublishClaim()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-auth-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string managedPath = Path.Combine(tempRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
        string runtimePath = Path.Combine(tempRoot, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg");
        string releaseNotesPath = Path.Combine(tempRoot, "release-notes.md");
        string rollbackPlanPath = Path.Combine(tempRoot, "rollback.md");

        try
        {
            CreateMinimalNupkg(managedPath, "JYPPX.TensorRT.CSharp.API");
            CreateMinimalNupkg(runtimePath, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22");
            File.WriteAllText(releaseNotesPath, "# Release notes");
            File.WriteAllText(rollbackPlanPath, "# Rollback plan");

            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublishAuthorizationInputTemplate.ps1"));
            using JsonDocument templateDocument = ReadFinalReleaseJson("owner-publish-authorization-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["ownerName"] = "Release Owner";
            values["ownerDecisionTimestampUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["authorizationDecision"] = "approved-for-owner-run";
            values["authorizedRoutes"] = new[] { "nuget-small-bridge-core", "github-packages-full-runtime" };
            values["managedPackageVersion"] = "4.0.0";
            values["runtimePackageVersion"] = "4.0.0";
            values["managedNupkgPath"] = managedPath;
            values["managedNupkgSha256"] = Sha256(managedPath);
            values["runtimeNupkgPath"] = runtimePath;
            values["runtimeNupkgSha256"] = Sha256(runtimePath);
            values["releaseNotesPath"] = releaseNotesPath;
            values["releaseNotesSha256"] = Sha256(releaseNotesPath);
            values["rollbackPlanPath"] = rollbackPlanPath;
            values["rollbackPlanSha256"] = Sha256(rollbackPlanPath);
            values["confirmsNoTokenPersisted"] = "true";
            values["confirmsNoDryRunArtifactSubstitution"] = "true";
            values["confirmsPackageHashesReviewed"] = "true";
            values["confirmsPublishCommandReviewed"] = "true";
            values["confirmsPublicPackageDownloadProofStillRequired"] = "true";

            string readyPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-publish-authorization-input.ready.json");
            File.WriteAllText(readyPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublishAuthorizationInput.ps1"),
                "-InputPath",
                "artifacts/final-release/owner-publish-authorization-input.ready.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("owner-publish-authorization-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("owner-publish-authorization-input-ready-for-owner-run", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(validation.GetProperty("ownerPublishAuthorizationReady").GetBoolean());
            Assert.False(validation.GetProperty("performsPublish").GetBoolean());
            Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
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
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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

    private static string SixtyFour(string value)
    {
        return string.Concat(Enumerable.Repeat(value, 64));
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
