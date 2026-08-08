using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerPublicPublishExecutionResultCandidateTests
{
    [Fact]
    public void DefaultCandidateValidationStaysBlockedWithoutBlockers()
    {
        string tempRoot = CreateTempRoot();
        try
        {
            RunPowerShell("Test-OwnerPublicPublishExecutionResultPreflight.ps1", tempRoot, "-Strict");
            RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1", tempRoot);
            RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", tempRoot, "-Strict");

            using JsonDocument document = ReadFinalReleaseJson(tempRoot, "owner-public-publish-execution-result-candidate-validation.json");
            JsonElement validation = document.RootElement;
            Assert.Equal("owner-public-publish-execution-result-candidate-validation", validation.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-owner-public-publish-execution-result-input-required", validation.GetProperty("validationState").GetString());
            Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
            Assert.Equal(0, validation.GetProperty("candidateItemCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
            Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
            AssertValidationItem(validation, "owner-input-present", passed: false);
            AssertValidationItem(validation, "forbidden-substitutes-absent", passed: true);
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    [Fact]
    public void ReadyOwnerInputValidatesPublicPublishCandidateWithoutPublishingClaims()
    {
        string tempRoot = CreateTempRoot();
        try
        {
            RunPowerShell("Export-OwnerPublicPublishExecutionResultInputContract.ps1", tempRoot);
            RunPowerShell("Export-OwnerPublicPublishExecutionResultInputTemplate.ps1", tempRoot);
            WriteReadyGitHubActionsRunEvidenceValidation(tempRoot);
            string inputPath = FillOwnerInputTemplate(tempRoot);

            RunPowerShell("Test-OwnerPublicPublishExecutionResultPreflight.ps1", tempRoot, "-InputPath", inputPath, "-Strict");
            RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1", tempRoot, "-InputPath", inputPath);
            RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", tempRoot, "-Strict");

            using JsonDocument candidateDocument = ReadFinalReleaseJson(tempRoot, "owner-public-publish-execution-result-candidate.json");
            JsonElement candidate = candidateDocument.RootElement;
            Assert.Equal(1, candidate.GetProperty("candidateItemCount").GetInt32());
            Assert.Equal("72d65909a120e1e550568e01796bbbdb5ec2b2e4", candidate.GetProperty("sourceHeadSha").GetString());
            Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
            Assert.False(candidate.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());

            using JsonDocument validationDocument = ReadFinalReleaseJson(tempRoot, "owner-public-publish-execution-result-candidate-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("owner-public-publish-execution-result-candidate-ready", validation.GetProperty("validationState").GetString());
            Assert.True(validation.GetProperty("proofCandidateReady").GetBoolean());
            Assert.Equal(1, validation.GetProperty("candidateItemCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.Equal("JYPPX.TensorRT.CSharp.API", validation.GetProperty("publicPackageId").GetString());
            Assert.Equal("4.0.0", validation.GetProperty("publicPackageVersion").GetString());
            Assert.StartsWith("https://www.nuget.org/packages/", validation.GetProperty("publicPackageUrl").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("https://github.com/", validation.GetProperty("githubReleaseAssetUrl").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.False(validation.GetProperty("performsPublish").GetBoolean());
            Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
            AssertValidationItem(validation, "source-github-actions-run-evidence-ready", passed: true);
            AssertValidationItem(validation, "public-package-url-nuget", passed: true);
            AssertValidationItem(validation, "github-release-asset-url", passed: true);
            AssertValidationItem(validation, "forbidden-substitutes-absent", passed: true);
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    [Fact]
    public void ForbiddenSubstitutesBecomeBlockers()
    {
        string tempRoot = CreateTempRoot();
        try
        {
            Directory.CreateDirectory(tempRoot);
            string candidatePath = Path.Combine(tempRoot, "owner-public-publish-execution-result-candidate.json");
            File.WriteAllText(
                candidatePath,
                """
                {
                  "recordKind": "owner-public-publish-execution-result-candidate",
                  "candidateState": "owner-public-publish-execution-result-candidate-imported",
                  "readyCandidateCount": 1,
                  "candidateItemCount": 1,
                  "blockedCandidateCount": 0,
                  "performsPublish": false,
                  "usesPublishToken": false,
                  "canPublishPublicly": false,
                  "canCloseReleaseIssue": false,
                  "canPromoteRuntimeProof": false,
                  "isReleaseReady": false,
                  "isRuntimeExecutionProof": false,
                  "isPostPublishProof": false,
                  "isReleaseCloseProof": false,
                  "boundary": "not runtime proof; not post-publish proof; cannot close",
                  "candidateItems": [
                    {
                      "fieldValues": {
                        "publicPackageSource": "local feed",
                        "publicPackageUrl": "file://local-feed/JYPPX.TensorRT.CSharp.API.4.0.0.nupkg",
                        "ownerReviewer": "manual approval"
                      },
                      "resultSummary": {
                        "sourceGitHubActionsRunEvidenceReady": true,
                        "sourceGitHubActionsRunId": "29160655818",
                        "sourceGitHubActionsRunUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29160655818",
                        "sourceHeadSha": "72d65909a120e1e550568e01796bbbdb5ec2b2e4",
                        "sourceWorkflowRunLogSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "sourceArtifactManifestSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "publicPackageId": "JYPPX.TensorRT.CSharp.API",
                        "publicPackageVersion": "4.0.0",
                        "publicPackageSource": "local feed",
                        "publicPackageUrl": "file://local-feed/JYPPX.TensorRT.CSharp.API.4.0.0.nupkg",
                        "publicPackageSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                        "publicPackagePublishedAtUtc": "2026-07-13T03:00:00Z",
                        "managedPackageId": "JYPPX.TensorRT.CSharp.API",
                        "runtimePackageId": "JYPPX.TensorRT.CSharp.API.runtime.win-x64",
                        "managedPackageUrl": "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0",
                        "runtimePackageUrl": "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.runtime.win-x64/4.0.0",
                        "managedPackageSha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                        "runtimePackageSha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                        "githubReleaseUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/tag/v4.0.0",
                        "githubReleaseAssetUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/download/v4.0.0/runtime.zip",
                        "githubReleaseAssetSha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                        "packageManagedPackageSourceChannel": "nuget.org",
                        "packageSourceChannel": "nuget.org+github",
                        "ownerReviewer": "manual approval",
                        "ownerReviewTimestampUtc": "2026-07-13T03:10:00Z",
                        "ownerAuthorizationId": "owner-authorization-001",
                        "ownerAuthorizationTimestampUtc": "2026-07-13T02:55:00Z"
                      }
                    }
                  ]
                }
                """);

            RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", tempRoot, "-InputPath", candidatePath);

            using JsonDocument document = ReadFinalReleaseJson(tempRoot, "owner-public-publish-execution-result-candidate-validation.json");
            JsonElement validation = document.RootElement;
            Assert.Equal("invalid-owner-public-publish-execution-result-candidate", validation.GetProperty("validationState").GetString());
            Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
            Assert.True(validation.GetProperty("failedBlockerCount").GetInt32() > 0);
            AssertValidationItem(validation, "forbidden-substitutes-absent", passed: false);
            string[] findings = validation.GetProperty("forbiddenSubstituteFindings").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains("local-feed", findings);
            Assert.Contains("manual-approval", findings);
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    private static string FillOwnerInputTemplate(string outputRoot)
    {
        string templatePath = Path.Combine(outputRoot, "owner-public-publish-execution-result-input-template.json");
        JsonObject root = JsonNode.Parse(File.ReadAllText(templatePath))!.AsObject();
        JsonArray fields = root["ownerInputFields"]!.AsArray();
        foreach (JsonNode? node in fields)
        {
            JsonObject field = node!.AsObject();
            string name = field["name"]!.GetValue<string>();
            field["value"] = ValueForField(name);
            field["valueState"] = "owner-input-ready-for-candidate";
            field["ready"] = true;
        }

        string inputPath = Path.Combine(outputRoot, "owner-public-publish-execution-result-owner-ready.json");
        File.WriteAllText(inputPath, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return inputPath;
    }

    private static string ValueForField(string name)
    {
        if (name.EndsWith("Sha256", StringComparison.Ordinal)) return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        if (name.EndsWith("ExitCode", StringComparison.Ordinal)) return "0";
        if (name.EndsWith("AtUtc", StringComparison.Ordinal) || name.EndsWith("TimestampUtc", StringComparison.Ordinal)) return "2026-07-13T03:00:00Z";
        if (name.EndsWith("Email", StringComparison.Ordinal)) return "owner@example.com";
        if (name.StartsWith("no", StringComparison.OrdinalIgnoreCase)) return "confirmed";
        if (name.Contains("Confirmation", StringComparison.OrdinalIgnoreCase)) return "confirmed";
        if (name.Contains("WasNot", StringComparison.OrdinalIgnoreCase)) return "not-applicable-owner-evidence";
        if (name.Contains("Explanation", StringComparison.OrdinalIgnoreCase)) return "not-applicable-owner-evidence";
        if (name.Contains("Decision", StringComparison.OrdinalIgnoreCase)) return "approved";
        if (name.Contains("Reason", StringComparison.OrdinalIgnoreCase)) return "owner-recorded-reason";
        if (name.Contains("Status", StringComparison.OrdinalIgnoreCase)) return "completed";
        if (name.Contains("Command", StringComparison.OrdinalIgnoreCase)) return "dotnet nuget push package --source https://api.nuget.org/v3/index.json";
        if (name.EndsWith("Path", StringComparison.Ordinal)) return $@"E:\owner-evidence\{name}.txt";
        if (name.EndsWith("Url", StringComparison.Ordinal)) return UrlForField(name);

        return name switch
        {
            "publicPackageId" => "JYPPX.TensorRT.CSharp.API",
            "managedPackageId" => "JYPPX.TensorRT.CSharp.API",
            "runtimePackageId" => "JYPPX.TensorRT.CSharp.API.runtime.win-x64",
            "packageManagedPackageId" => "JYPPX.TensorRT.CSharp.API",
            "packageNativeBridgePackageId" => "JYPPX.TensorRT.CSharp.API.NativeBridge",
            "packageRuntimePackageId" => "JYPPX.TensorRT.CSharp.API.runtime.win-x64",
            "publicPackageVersion" => "4.0.0",
            "packageManagedPackageVersion" => "4.0.0",
            "packageNativeBridgePackageVersion" => "4.0.0",
            "packageRuntimePackageVersion" => "4.0.0",
            "publicPackageSource" => "nuget.org",
            "nugetPushSource" => "https://api.nuget.org/v3/index.json",
            "cleanConsumerRestorePackageSource" => "https://api.nuget.org/v3/index.json",
            "packageManagedPackageSourceChannel" => "nuget.org",
            "packageSourceChannel" => "nuget.org+github",
            "publicPackageVisibility" => "public",
            "publicPackageOwnerAccount" => "guojin-yan",
            "ownerReviewer" => "guojin-yan",
            "ownerReviewerEmail" => "owner@example.com",
            "ownerAuthorizationId" => "owner-authorization-001",
            "ownerApprovalId" => "owner-approval-001",
            "ownerSignature" => "guojin-yan",
            _ => $"owner-evidence-{name}"
        };
    }

    private static string UrlForField(string name)
    {
        if (name.Contains("githubReleaseAsset", StringComparison.OrdinalIgnoreCase))
        {
            return "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/download/v4.0.0/tensorrt-csharp-api-runtime.zip";
        }

        if (name.Contains("githubRelease", StringComparison.OrdinalIgnoreCase))
        {
            return "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/tag/v4.0.0";
        }

        if (name.Contains("runtime", StringComparison.OrdinalIgnoreCase))
        {
            return "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.runtime.win-x64/4.0.0";
        }

        if (name.Contains("nativeBridge", StringComparison.OrdinalIgnoreCase))
        {
            return "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.NativeBridge/4.0.0";
        }

        return "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0";
    }

    private static void WriteReadyGitHubActionsRunEvidenceValidation(string outputRoot)
    {
        File.WriteAllText(
            Path.Combine(outputRoot, "github-actions-run-evidence-import-validation.json"),
            """
            {
              "recordKind": "github-actions-run-evidence-import-validation",
              "validationState": "github-actions-run-evidence-ready",
              "githubActionsRunEvidenceReady": true,
              "runId": "29160655818",
              "runUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29160655818",
              "headSha": "72d65909a120e1e550568e01796bbbdb5ec2b2e4",
              "workflowRunLogSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "artifactManifestSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
              "performsPublish": false,
              "usesPublishToken": false,
              "canPublishPublicly": false,
              "canCloseReleaseIssue": false
            }
            """);
    }

    private static JsonDocument ReadFinalReleaseJson(string outputRoot, string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, fileName)));
    }

    private static void AssertValidationItem(JsonElement validation, string id, bool passed)
    {
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == passed);
    }

    private static string CreateTempRoot()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-public-publish-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        return tempRoot;
    }

    private static void DeleteTempRoot(string tempRoot)
    {
        if (Directory.Exists(tempRoot))
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static string RunPowerShell(string scriptName, string outputRoot, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        process.StartInfo.ArgumentList.Add("-OutputDirectory");
        process.StartInfo.ArgumentList.Add(outputRoot);
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

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
