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
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-auth-blocked-" + Guid.NewGuid().ToString("N"));
        string blockedReadinessMatrixPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.blocked.json");

        try
        {
            Directory.CreateDirectory(tempRoot);
            WriteReadinessMatrix(blockedReadinessMatrixPath, ready: false);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublishAuthorizationInputTemplate.ps1"),
                "-PreReleaseReadinessMatrixPath",
                blockedReadinessMatrixPath);
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublishAuthorizationInput.ps1"), "-Strict");

            using JsonDocument templateDocument = ReadFinalReleaseJson("owner-publish-authorization-input.template.json");
            JsonElement template = templateDocument.RootElement;
            Assert.Equal("owner-publish-authorization-input", template.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-owner-publish-authorization-required", template.GetProperty("validationState").GetString());
            Assert.Equal("owner-authorization-required", template.GetProperty("authorizationDecision").GetString());
            Assert.Equal("manual-owner-run-only", template.GetProperty("ownerAuthorizationScope").GetString());
            Assert.True(template.GetProperty("publishTargetChannels").GetArrayLength() >= 2);
            Assert.True(template.TryGetProperty("ownerAuthorizationId", out _));
            Assert.True(template.TryGetProperty("publishCommandPlanSha256", out _));
            Assert.True(template.TryGetProperty("managedPublishCommandSha256", out _));
            Assert.True(template.TryGetProperty("runtimePublishCommandSha256", out _));
            Assert.True(template.TryGetProperty("sourceRunnerQueueStatus", out _));
            Assert.True(template.TryGetProperty("sourceRunnerInfrastructureStatus", out _));
            Assert.True(template.TryGetProperty("sourceRunnerOwnerAction", out _));
            Assert.Equal("blocked-real-public-package-and-runtime-proof-required", template.GetProperty("preReleaseReadinessMatrixState").GetString());
            Assert.False(template.GetProperty("preReleaseReadinessMatrixReady").GetBoolean());
            Assert.Equal(4, template.GetProperty("preReleaseBlockedLaneCount").GetInt32());
            Assert.Equal(4, template.GetProperty("preReleaseBlockedLanes").GetArrayLength());
            Assert.True(template.TryGetProperty("confirmsPreReleaseReadinessMatrixReviewed", out _));
            Assert.False(template.GetProperty("performsPublish").GetBoolean());
            Assert.False(template.GetProperty("usesPublishToken").GetBoolean());
            Assert.True(template.GetProperty("requiresOwnerAuthorization").GetBoolean());
            Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(template.GetProperty("isPostPublishProof").GetBoolean());
            Assert.Contains("dotnet nuget push", template.GetProperty("publishCommandTemplates").EnumerateArray().First().GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("does not execute dotnet nuget push", template.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("queued GitHub Actions run", template.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("missing self-hosted runner", template.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("--force publish", template.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()), StringComparer.OrdinalIgnoreCase);

            using JsonDocument validationDocument = ReadFinalReleaseJson("owner-publish-authorization-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("owner-publish-authorization-input-validation", validation.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-owner-publish-authorization-required", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
            Assert.False(validation.GetProperty("ownerPublishAuthorizationReady").GetBoolean());
            Assert.Equal("manual-owner-run-only", validation.GetProperty("ownerAuthorizationScope").GetString());
            Assert.Equal(2, validation.GetProperty("publishTargetChannelCount").GetInt32());
            Assert.Equal("blocked-real-public-package-and-runtime-proof-required", validation.GetProperty("preReleaseReadinessMatrixState").GetString());
            Assert.False(validation.GetProperty("preReleaseReadinessMatrixReady").GetBoolean());
            Assert.Equal(4, validation.GetProperty("preReleaseReadinessBlockedLaneCount").GetInt32());
            Assert.Empty(validation.GetProperty("preReleaseMissingLaneIds").EnumerateArray());
            Assert.Empty(validation.GetProperty("preReleaseMetadataMissingLaneIds").EnumerateArray());
            Assert.Empty(validation.GetProperty("preReleasePrematurePromoteFindings").EnumerateArray());
            AssertValidationItemFailed(validation, "source-runner-not-queued");
            AssertValidationItemFailed(validation, "source-runner-infrastructure-ready");
            AssertValidationItemFailed(validation, "pre-release-readiness-ready-for-publish-authorization");
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

    [Fact]
    public void OwnerPublishAuthorizationValidatorRejectsPersistedTokenAndDryRunArtifacts()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublishAuthorizationInputTemplate.ps1"));
        using JsonDocument templateDocument = ReadFinalReleaseJson("owner-publish-authorization-input.template.json");
        Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
        values["ownerName"] = "ghp_abcdefghijklmnopqrstuvwxyz123456";
        values["authorizationDecision"] = "approved-for-owner-run";
        values["authorizedRoutes"] = new[] { "nuget-small-bridge-core" };
        values["publishTargetChannels"] = new[] { "nuget-small-bridge-core" };
        values["ownerAuthorizationId"] = "owner-auth-20260712-001";
        values["ownerAuthorizationScope"] = "manual-owner-run-only";
        values["managedNupkgPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "29162977180", "package-managed-dry-run", "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
        values["runtimeNupkgPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "29162977180", "package-managed-dry-run", "runtime.nupkg");
        values["releaseNotesPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "release-notes.md");
        values["rollbackPlanPath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "rollback.md");
        values["managedNupkgSha256"] = SixtyFour("a");
        values["runtimeNupkgSha256"] = SixtyFour("b");
        values["releaseNotesSha256"] = SixtyFour("c");
        values["rollbackPlanSha256"] = SixtyFour("d");
        values["publishCommandPlanSha256"] = SixtyFour("e");
        values["managedPublishCommandSha256"] = SixtyFour("f");
        values["runtimePublishCommandSha256"] = SixtyFour("0");
        values["sourceRunnerQueueStatus"] = "queued";
        values["sourceRunnerInfrastructureStatus"] = "missing-self-hosted-runner";
        values["sourceRunnerOwnerAction"] = "owner-infra-action-required-until-runner-completed-and-available";
        values["ownerDecisionTimestampUtc"] = DateTimeOffset.UtcNow.ToString("O");
        values["managedPackageVersion"] = "4.0.0";
        values["runtimePackageVersion"] = "4.0.0";
        values["confirmsNoTokenPersisted"] = "true";
        values["confirmsPreReleaseReadinessMatrixReviewed"] = "true";
        values["confirmsNoDryRunArtifactSubstitution"] = "true";
        values["confirmsPackageHashesReviewed"] = "true";
        values["confirmsPublishCommandReviewed"] = "true";
        values["confirmsPublishCommandHashesReviewed"] = "true";
        values["confirmsNoForcePublish"] = "true";
        values["confirmsNoQueuedRunOrMissingRunnerSubstitution"] = "true";
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
        AssertValidationItemFailed(validation, "source-runner-not-queued");
        AssertValidationItemFailed(validation, "source-runner-infrastructure-ready");
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
        string readinessMatrixPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.ready.json");

        try
        {
            CreateMinimalNupkg(managedPath, "JYPPX.TensorRT.CSharp.API");
            CreateMinimalNupkg(runtimePath, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22");
            File.WriteAllText(releaseNotesPath, "# Release notes");
            File.WriteAllText(rollbackPlanPath, "# Rollback plan");
            WriteReadinessMatrix(readinessMatrixPath, ready: true);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublishAuthorizationInputTemplate.ps1"),
                "-PreReleaseReadinessMatrixPath",
                readinessMatrixPath);
            using JsonDocument templateDocument = ReadFinalReleaseJson("owner-publish-authorization-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["ownerName"] = "Release Owner";
            values["ownerDecisionTimestampUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["authorizationDecision"] = "approved-for-owner-run";
            values["authorizedRoutes"] = new[] { "nuget-small-bridge-core", "github-packages-bridge" };
            values["publishTargetChannels"] = new[] { "nuget-small-bridge-core", "github-packages-bridge" };
            values["ownerAuthorizationId"] = "owner-auth-20260712-001";
            values["ownerAuthorizationScope"] = "manual-owner-run-only";
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
            values["publishCommandPlanPath"] = releaseNotesPath;
            values["publishCommandPlanSha256"] = Sha256(releaseNotesPath);
            values["managedPublishCommandSha256"] = SixtyFour("e");
            values["runtimePublishCommandSha256"] = SixtyFour("f");
            values["sourceRunnerQueueStatus"] = "completed";
            values["sourceRunnerInfrastructureStatus"] = "available";
            values["sourceRunnerOwnerAction"] = "owner-infra-action-reviewed-and-clear";
            values["confirmsPreReleaseReadinessMatrixReviewed"] = "true";
            values["confirmsNoTokenPersisted"] = "true";
            values["confirmsNoDryRunArtifactSubstitution"] = "true";
            values["confirmsPackageHashesReviewed"] = "true";
            values["confirmsPublishCommandReviewed"] = "true";
            values["confirmsPublishCommandHashesReviewed"] = "true";
            values["confirmsNoForcePublish"] = "true";
            values["confirmsNoQueuedRunOrMissingRunnerSubstitution"] = "true";
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
            Assert.Equal("manual-owner-run-only", validation.GetProperty("ownerAuthorizationScope").GetString());
            Assert.Equal(2, validation.GetProperty("publishTargetChannelCount").GetInt32());
            Assert.Equal(2, validation.GetProperty("authorizedRouteCount").GetInt32());
            Assert.Equal("completed", validation.GetProperty("sourceRunnerQueueStatus").GetString());
            Assert.Equal("available", validation.GetProperty("sourceRunnerInfrastructureStatus").GetString());
            Assert.Equal("pre-release-package-proof-ready", validation.GetProperty("preReleaseReadinessMatrixState").GetString());
            Assert.True(validation.GetProperty("preReleaseReadinessMatrixReady").GetBoolean());
            Assert.Equal(0, validation.GetProperty("preReleaseReadinessBlockedLaneCount").GetInt32());
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
            static property => ToObject(property.Value));
    }

    private static object? ToObject(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.Object => element.EnumerateObject().ToDictionary(
                static property => property.Name,
                static property => ToObject(property.Value)),
            JsonValueKind.Array => element.EnumerateArray().Select(static item => ToObject(item)).ToArray(),
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number => element.TryGetInt64(out long longValue) ? longValue : element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            _ => element.ToString()
        };
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

    private static void WriteReadinessMatrix(string path, bool ready)
    {
        List<Dictionary<string, object?>> lanes =
        [
            ReadinessLane("source-quality-ci", ready),
            ReadinessLane("current-head-package-dry-run", ready),
            ReadinessLane("owner-dispatch-pack", ready),
            ReadinessLane("public-package-download", ready),
            ReadinessLane("clean-external-package-consumer-runtime", ready),
            ReadinessLane("post-publish-clean-consumer-proof", ready),
        ];

        int readyLaneCount = lanes.Count(static lane => (bool)lane["ready"]!);
        var matrix = new Dictionary<string, object?>
        {
            ["recordKind"] = "pre-release-package-proof-readiness-matrix",
            ["generatedAtUtc"] = DateTimeOffset.UtcNow.ToString("O"),
            ["matrixState"] = ready ? "pre-release-package-proof-ready" : "blocked-real-public-package-and-runtime-proof-required",
            ["currentHead"] = "8065fa6e8adb66177522cb535f981c80d5f793d4",
            ["sourceQualityRunId"] = "29235831169",
            ["readyLaneCount"] = readyLaneCount,
            ["blockedLaneCount"] = lanes.Count - readyLaneCount,
            ["currentHeadPackageDryRunReady"] = ready,
            ["ownerDispatchPackReadyForOwner"] = ready,
            ["publicPackageDownloadProofReady"] = ready,
            ["packageConsumerRuntimeProofReady"] = ready,
            ["postPublishProofReady"] = ready,
            ["performsPublish"] = false,
            ["usesPublishToken"] = false,
            ["canPublishPublicly"] = false,
            ["canCloseReleaseIssue"] = false,
            ["canPromoteProof"] = false,
            ["lanes"] = lanes,
            ["safetyBoundary"] = "Test fixture only; no publish, no token use, and no release close side effects.",
        };

        File.WriteAllText(path, JsonSerializer.Serialize(matrix, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static Dictionary<string, object?> ReadinessLane(string id, bool ready)
    {
        bool laneReady = ready || id is "source-quality-ci" or "owner-dispatch-pack";
        return new Dictionary<string, object?>
        {
            ["id"] = id,
            ["title"] = id,
            ["state"] = laneReady ? "ready" : "blocked-fixture-proof-required",
            ["ready"] = laneReady,
            ["sourceArtifact"] = $"artifacts/final-release/{id}.json",
            ["requiredEvidence"] = $"Fixture required evidence for {id}.",
            ["requiredProof"] = $"Fixture required evidence for {id}.",
            ["blockedReason"] = laneReady ? "none" : "fixture blocked until real public package/runtime/post-publish proof exists",
            ["validatorPath"] = $"eng\\Test-{id}.ps1 -Strict",
            ["failedBlockerCount"] = 0,
            ["failedActionRequiredCount"] = laneReady ? 0 : 1,
            ["performsPublish"] = false,
            ["canPromotePublicProof"] = false,
            ["canPromoteRuntimeProof"] = false,
            ["canPromotePostPublishProof"] = false,
            ["canPromoteProof"] = false,
            ["canPublishPublicly"] = false,
            ["canCloseReleaseIssue"] = false,
            ["isPackageConsumerRuntimeProof"] = false,
            ["isPostPublishProof"] = false,
        };
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
