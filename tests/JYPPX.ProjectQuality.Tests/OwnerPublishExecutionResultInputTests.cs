using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPublishExecutionResultInputTests
{
    [Fact]
    public void OwnerPublishExecutionResultTemplateExportsBlockedSideEffectFreeSurface()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-publish-template-" + Guid.NewGuid().ToString("N"));
        string outputRoot = Path.Combine(tempRoot, "final-release");
        string readinessMatrixPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.blocked.json");

        try
        {
            Directory.CreateDirectory(tempRoot);
            WriteReadinessMatrix(readinessMatrixPath, ready: false);

            RunPowerShell(
                "Export-OwnerPublishExecutionResultInputTemplate.ps1",
                "-OutputRoot",
                outputRoot,
                "-PreReleaseReadinessMatrixPath",
                readinessMatrixPath);
            RunPowerShell(
                "Test-OwnerPublishExecutionResultInput.ps1",
                "-InputPath",
                Path.Combine(outputRoot, "owner-publish-execution-result-input.template.json"),
                "-OutputRoot",
                outputRoot,
                "-Strict");

            using JsonDocument templateDocument = ReadJson(outputRoot, "owner-publish-execution-result-input.template.json");
            JsonElement template = templateDocument.RootElement;
            Assert.Equal("owner-publish-execution-result-input", template.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-owner-publish-execution-result-required", template.GetProperty("validationState").GetString());
            Assert.False(template.GetProperty("ownerExecutionResultReady").GetBoolean());
            Assert.False(template.GetProperty("performsPublish").GetBoolean());
            Assert.False(template.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(template.GetProperty("isPostPublishProof").GetBoolean());
            Assert.Equal("blocked-real-public-package-and-runtime-proof-required", template.GetProperty("preReleaseReadinessMatrixState").GetString());
            Assert.False(template.GetProperty("preReleaseReadinessMatrixReady").GetBoolean());
            Assert.Equal(4, template.GetProperty("preReleaseBlockedLaneCount").GetInt32());
            Assert.Equal(4, template.GetProperty("preReleaseBlockedLanes").GetArrayLength());
            Assert.True(template.TryGetProperty("confirmsPreReleaseReadinessMatrixReviewed", out _));

            using JsonDocument validationDocument = ReadJson(outputRoot, "owner-publish-execution-result-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("owner-publish-execution-result-input-validation", validation.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-owner-publish-execution-result-required", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
            Assert.False(validation.GetProperty("ownerExecutionResultReady").GetBoolean());
            Assert.Equal("blocked-real-public-package-and-runtime-proof-required", validation.GetProperty("preReleaseReadinessMatrixState").GetString());
            Assert.False(validation.GetProperty("preReleaseReadinessMatrixReady").GetBoolean());
            Assert.Equal(4, validation.GetProperty("preReleaseReadinessBlockedLaneCount").GetInt32());
            AssertValidationItemFailed(validation, "pre-release-readiness-ready-for-owner-publish-execution");
            AssertFalseProofPublishCloseFlags(validation);
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
    public void OwnerPublishExecutionResultValidatorRejectsTokenAndSubstituteEvidence()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-publish-bad-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string transcriptPath = Path.Combine(tempRoot, "push-transcript.log");
        string stdoutPath = Path.Combine(tempRoot, "push-stdout.log");
        string stderrPath = Path.Combine(tempRoot, "push-stderr.log");
        string notesPath = Path.Combine(tempRoot, "release-notes.md");
        string rollbackPath = Path.Combine(tempRoot, "rollback.md");
        string dryRunPackagePath = Path.Combine(tempRoot, "package-managed-dry-run", "fake.nupkg");
        string outputRoot = Path.Combine(tempRoot, "final-release");
        string readinessMatrixPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.blocked.json");

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(dryRunPackagePath)!);
            WriteReadinessMatrix(readinessMatrixPath, ready: false);
            File.WriteAllText(transcriptPath, "dotnet nuget push --api-key ghp_abcdefghijklmnopqrstuvwxyz0123456789");
            File.WriteAllText(stdoutPath, "published using NUGET_AUTH_TOKEN");
            File.WriteAllText(stderrPath, string.Empty);
            File.WriteAllText(notesPath, "# notes");
            File.WriteAllText(rollbackPath, "# rollback");
            File.WriteAllText(dryRunPackagePath, "dry run package");

            RunPowerShell(
                "Export-OwnerPublishExecutionResultInputTemplate.ps1",
                "-OutputRoot",
                outputRoot,
                "-PreReleaseReadinessMatrixPath",
                readinessMatrixPath);
            using JsonDocument templateDocument = ReadJson(outputRoot, "owner-publish-execution-result-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["managedPackageVersion"] = "4.0.0";
            values["runtimePackageVersion"] = "4.0.0";
            values["ownerName"] = "Release Owner";
            values["ownerReviewedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["ownerApprovalReference"] = "approval-001";
            values["publishCommandReviewed"] = "true";
            values["publishExecutedByOwner"] = "true";
            values["publishStartedAtUtc"] = DateTimeOffset.UtcNow.AddMinutes(-2).ToString("O");
            values["publishCompletedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["publishExitCode"] = "0";
            values["publishCommand"] = "dotnet nuget push package.nupkg --api-key ghp_badtoken";
            values["pushTranscriptPath"] = transcriptPath;
            values["pushTranscriptSha256"] = Sha256(transcriptPath);
            values["pushStdoutPath"] = stdoutPath;
            values["pushStdoutSha256"] = Sha256(stdoutPath);
            values["pushStderrPath"] = stderrPath;
            values["pushStderrSha256"] = Sha256(stderrPath);
            values["publicManagedPackageUrl"] = "file:///local/package";
            values["publicRuntimePackageUrl"] = "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/123/artifacts/package-managed-dry-run";
            values["nugetPackageMetadataUrl"] = "https://api.nuget.org/v3/index.json";
            values["githubPackagesMetadataUrl"] = "https://nuget.pkg.github.com/guojin-yan/index.json";
            values["downloadedManagedNupkgPath"] = dryRunPackagePath;
            values["downloadedManagedNupkgSha256"] = Sha256(dryRunPackagePath);
            values["downloadedRuntimeNupkgPath"] = dryRunPackagePath;
            values["downloadedRuntimeNupkgSha256"] = Sha256(dryRunPackagePath);
            values["releaseNotesPath"] = notesPath;
            values["releaseNotesSha256"] = Sha256(notesPath);
            values["rollbackPlanPath"] = rollbackPath;
            values["rollbackPlanSha256"] = Sha256(rollbackPath);
            values["rollbackDecision"] = "rollback-plan-reviewed";
            values["confirmsPreReleaseReadinessMatrixReviewed"] = "true";

            string misusePath = Path.Combine(outputRoot, "owner-publish-execution-result-input.misuse.json");
            File.WriteAllText(misusePath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                "Test-OwnerPublishExecutionResultInput.ps1",
                "-InputPath",
                misusePath,
                "-OutputRoot",
                outputRoot,
                "-Strict");

            using JsonDocument validationDocument = ReadJson(outputRoot, "owner-publish-execution-result-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("blocked-owner-publish-execution-result-required", validation.GetProperty("validationState").GetString());
            Assert.False(validation.GetProperty("ownerExecutionResultReady").GetBoolean());
            AssertValidationItemFailed(validation, "publish-command-shape");
            AssertValidationItemFailed(validation, "publicManagedPackageUrl-https-public");
            AssertValidationItemFailed(validation, "publicRuntimePackageUrl-https-public");
            AssertValidationItemFailed(validation, "downloadedManagedNupkg-path-not-forbidden-substitute");
            AssertValidationItemFailed(validation, "pushTranscriptPath-no-token-like-content");
            AssertValidationItemFailed(validation, "pushStdoutPath-no-token-like-content");
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
    public void OwnerPublishExecutionResultValidatorAcceptsReadyShapeWithoutClosingReleaseIssue()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-publish-good-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string transcriptPath = Path.Combine(tempRoot, "push-transcript.log");
        string stdoutPath = Path.Combine(tempRoot, "push-stdout.log");
        string stderrPath = Path.Combine(tempRoot, "push-stderr.log");
        string managedPackagePath = Path.Combine(tempRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
        string runtimePackagePath = Path.Combine(tempRoot, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg");
        string notesPath = Path.Combine(tempRoot, "release-notes.md");
        string rollbackPath = Path.Combine(tempRoot, "rollback.md");
        string outputRoot = Path.Combine(tempRoot, "final-release");
        string readinessMatrixPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.ready.json");

        try
        {
            WriteReadinessMatrix(readinessMatrixPath, ready: true);
            File.WriteAllText(transcriptPath, "dotnet nuget push JYPPX.TensorRT.CSharp.API.4.0.0.nupkg --api-key REDACTED --source https://api.nuget.org/v3/index.json");
            File.WriteAllText(stdoutPath, "owner publish completed");
            File.WriteAllText(stderrPath, string.Empty);
            File.WriteAllText(managedPackagePath, "managed package from public source");
            File.WriteAllText(runtimePackagePath, "runtime package from public source");
            File.WriteAllText(notesPath, "# release notes");
            File.WriteAllText(rollbackPath, "# rollback plan");

            RunPowerShell(
                "Export-OwnerPublishExecutionResultInputTemplate.ps1",
                "-OutputRoot",
                outputRoot,
                "-PreReleaseReadinessMatrixPath",
                readinessMatrixPath);
            using JsonDocument templateDocument = ReadJson(outputRoot, "owner-publish-execution-result-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["managedPackageVersion"] = "4.0.0";
            values["runtimePackageVersion"] = "4.0.0";
            values["ownerName"] = "Release Owner";
            values["ownerReviewedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["ownerApprovalReference"] = "approval-001";
            values["publishCommandReviewed"] = "true";
            values["publishExecutedByOwner"] = "true";
            values["publishStartedAtUtc"] = DateTimeOffset.UtcNow.AddMinutes(-2).ToString("O");
            values["publishCompletedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["publishExitCode"] = "0";
            values["publishCommand"] = "dotnet nuget push JYPPX.TensorRT.CSharp.API.4.0.0.nupkg --api-key REDACTED --source https://api.nuget.org/v3/index.json";
            values["pushTranscriptPath"] = transcriptPath;
            values["pushTranscriptSha256"] = Sha256(transcriptPath);
            values["pushStdoutPath"] = stdoutPath;
            values["pushStdoutSha256"] = Sha256(stdoutPath);
            values["pushStderrPath"] = stderrPath;
            values["pushStderrSha256"] = Sha256(stderrPath);
            values["publicManagedPackageUrl"] = "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0";
            values["publicRuntimePackageUrl"] = "https://nuget.pkg.github.com/guojin-yan/JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22/4.0.0";
            values["nugetPackageMetadataUrl"] = "https://api.nuget.org/v3/registration5-semver1/jyppx.tensorrt.csharp.api/index.json";
            values["githubPackagesMetadataUrl"] = "https://nuget.pkg.github.com/guojin-yan/index.json";
            values["downloadedManagedNupkgPath"] = managedPackagePath;
            values["downloadedManagedNupkgSha256"] = Sha256(managedPackagePath);
            values["downloadedRuntimeNupkgPath"] = runtimePackagePath;
            values["downloadedRuntimeNupkgSha256"] = Sha256(runtimePackagePath);
            values["releaseNotesPath"] = notesPath;
            values["releaseNotesSha256"] = Sha256(notesPath);
            values["rollbackPlanPath"] = rollbackPath;
            values["rollbackPlanSha256"] = Sha256(rollbackPath);
            values["rollbackDecision"] = "rollback-plan-reviewed";
            values["confirmsPreReleaseReadinessMatrixReviewed"] = "true";
            values["confirmsNoTokenPersisted"] = "true";
            values["confirmsNoTokenInTranscripts"] = "true";
            values["confirmsNoDryRunArtifactSubstitution"] = "true";
            values["confirmsNoLocalFeedSubstitution"] = "true";
            values["confirmsNoDirectNupkgSubstitution"] = "true";
            values["confirmsNoGitHubActionsArtifactSubstitution"] = "true";
            values["confirmsPublicPackageDownloadProofStillRequired"] = "true";
            values["confirmsPostPublishProofStillRequired"] = "true";

            string readyPath = Path.Combine(outputRoot, "owner-publish-execution-result-input.ready.json");
            File.WriteAllText(readyPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                "Test-OwnerPublishExecutionResultInput.ps1",
                "-InputPath",
                readyPath,
                "-OutputRoot",
                outputRoot,
                "-Strict");

            using JsonDocument validationDocument = ReadJson(outputRoot, "owner-publish-execution-result-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.True(
                validation.GetProperty("validationState").GetString() == "owner-publish-execution-result-input-ready",
                BuildFailedValidationSummary(validation));
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(validation.GetProperty("ownerExecutionResultReady").GetBoolean());
            Assert.Equal("pre-release-package-proof-ready", validation.GetProperty("preReleaseReadinessMatrixState").GetString());
            Assert.True(validation.GetProperty("preReleaseReadinessMatrixReady").GetBoolean());
            Assert.Equal(0, validation.GetProperty("preReleaseReadinessBlockedLaneCount").GetInt32());
            AssertFalseProofPublishCloseFlags(validation);
            Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
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

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        Assert.True(element.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static string BuildFailedValidationSummary(JsonElement validation)
    {
        string[] failedItems = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Where(static item => !item.GetProperty("passed").GetBoolean())
            .Select(static item => $"{item.GetProperty("id").GetString()}: {item.GetProperty("detail").GetString()}")
            .ToArray();

        return $"Expected ready validation, actual={validation.GetProperty("validationState").GetString()}, failedActionRequired={validation.GetProperty("failedActionRequiredCount").GetInt32()}, failedItems={string.Join("; ", failedItems)}";
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

    private static JsonDocument ReadJson(string outputRoot, string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, fileName)));
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream)).ToLowerInvariant();
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

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(
            process.ExitCode == 0,
            $"PowerShell script failed: {scriptName} {string.Join(' ', arguments)}{Environment.NewLine}STDOUT:{Environment.NewLine}{stdout}{Environment.NewLine}STDERR:{Environment.NewLine}{stderr}");

        return stdout;
    }
}
