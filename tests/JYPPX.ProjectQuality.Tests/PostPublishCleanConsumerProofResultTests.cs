using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishCleanConsumerProofResultTests
{
    [Fact]
    public void DefaultPostPublishProofResultImportRemainsBlockedAndFailOnNotProofRejectsIt()
    {
        RunPowerShell("Export-PostPublishCleanConsumerProofRecordContract.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofRecordContract.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerProofResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict");

        using JsonDocument contractDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-record-contract.json");
        JsonElement contract = contractDocument.RootElement;
        Assert.Equal("post-publish-clean-consumer-proof-record-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", contract.GetProperty("contractState").GetString());
        Assert.True(contract.GetProperty("requiredFieldCount").GetInt32() >= 40);
        AssertCanonicalContractFields(contract);

        using JsonDocument contractValidationDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-record-contract-validation.json");
        JsonElement contractValidation = contractValidationDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", contractValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, contractValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(contractValidation.GetProperty("canonicalRequiredFieldCount").GetInt32() >= 29);
        string contractDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "post-publish-clean-consumer-proof-record-contract.md"));
        Assert.Contains("cleanExternalConsumer.restoreLogSha256", contractDoc, StringComparison.Ordinal);
        Assert.Contains("hostMetadata.tensorRtLine", contractDoc, StringComparison.Ordinal);
        Assert.Contains("forbiddenSubstituteCounts.projectReferenceCount", contractDoc, StringComparison.Ordinal);
        Assert.Contains("blockedByDriverOnlyCount", contractDoc, StringComparison.Ordinal);

        using JsonDocument importDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-import.json");
        JsonElement import = importDocument.RootElement;

        Assert.Equal("post-publish-clean-consumer-proof-result-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-result-required", import.GetProperty("importState").GetString());
        Assert.False(import.GetProperty("proofCandidateReady").GetBoolean());
        Assert.True(import.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(import.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(import.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(import.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument candidateDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-proof-candidate", candidate.GetProperty("candidateState").GetString());
        Assert.False(candidate.GetProperty("proofCandidateReady").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());

        string failOutput = RunPowerShellExpectFailure("Test-PostPublishCleanConsumerProofResult.ps1", "-FailOnNotProof");
        Assert.Contains("not proof-ready", failOutput, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("post-publish-clean-consumer-proof-result-validation-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
    }

    [Fact]
    public void ReadyShapedPostPublishProofResultSetsOnlyProofCandidateReadyWithoutPromotingClassificationFlags()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-post-publish-proof-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string consumerRoot = Path.Combine(tempRoot, "clean-consumer");
        Directory.CreateDirectory(consumerRoot);
        string projectPath = Path.Combine(consumerRoot, "CleanConsumer.csproj");
        File.WriteAllText(projectPath, "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>");

        try
        {
            Dictionary<string, object?> values = new()
            {
                ["recordKind"] = "post-publish-clean-consumer-proof-result-owner-input",
                ["ownerInputState"] = "owner-filled-post-publish-clean-consumer-proof-result",
                ["publicPackageSourceUrl"] = "https://api.nuget.org/v3/index.json",
                ["publicPackageUrl"] = "https://api.nuget.org/v3/registration5-semver1/jyppx.tensorrt.csharp.api/index.json",
                ["publicPackageSourceKind"] = "nuget.org",
                ["managedPackageId"] = "JYPPX.TensorRT.CSharp.API",
                ["managedPackageVersion"] = "4.0.0",
                ["runtimePackageId"] = "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22",
                ["runtimePackageVersion"] = "4.0.0",
                ["runtimePackageKey"] = "win-x64-trt11.0-cuda13.2-cudnn9.22",
                ["cleanConsumerRoot"] = consumerRoot,
                ["consumerProjectPath"] = projectPath,
                ["restoreCommand"] = "dotnet restore --source https://api.nuget.org/v3/index.json",
                ["buildCommand"] = "dotnet build -c Release --no-restore",
                ["runCommand"] = "dotnet run -c Release --no-build",
                ["exitCode"] = 0,
                ["confirmsPostPublish"] = true,
                ["confirmsNotPrePublishSmoke"] = true,
                ["hostMetadata"] = new Dictionary<string, object?>
                {
                    ["os"] = "Windows 11",
                    ["arch"] = "x64",
                    ["rid"] = "win-x64",
                    ["gpuName"] = "NVIDIA RTX",
                    ["nvidiaDriver"] = "555.00",
                    ["cudaRuntimeToolkit"] = "13.2",
                    ["tensorrt"] = "11.0",
                    ["cudnn"] = "9.22",
                },
                ["ownerReviewer"] = "owner",
                ["ownerReviewedAtUtc"] = DateTimeOffset.UtcNow.ToString("O"),
            };

            string downloadedManagedPackageSha256 = AddEvidence(values, tempRoot, "downloadedManagedPackagePath", "downloadedManagedPackageSha256", "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
            AddEvidence(values, tempRoot, "downloadedRuntimePackagePath", "downloadedRuntimePackageSha256", "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg");
            AddEvidence(values, tempRoot, "installLogPath", "installLogSha256", "install.log");
            AddEvidence(values, tempRoot, "restoreLogPath", "restoreLogSha256", "restore.log");
            AddEvidence(values, tempRoot, "buildLogPath", "buildLogSha256", "build.log");
            AddEvidence(values, tempRoot, "runLogPath", "runLogSha256", "run.log");
            AddEvidence(values, tempRoot, "smokeStdoutPath", "smokeStdoutSha256", "stdout.log");
            AddEvidence(values, tempRoot, "smokeStderrPath", "smokeStderrSha256", "stderr.log");
            AddEvidence(values, tempRoot, "nativeAssetListingPath", "nativeAssetListingSha256", "native-assets.txt");
            AddEvidence(values, tempRoot, "dotnetInfoPath", "dotnetInfoSha256", "dotnet-info.txt");

            WriteReadySourceProofValidations(
                publicPackageUrl: "https://api.nuget.org/v3/registration5-semver1/jyppx.tensorrt.csharp.api/index.json",
                publicPackageVersion: "4.0.0",
                publicPackageSha256: downloadedManagedPackageSha256);

            string ownerInputPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-clean-consumer-proof-result.ready.json");
            File.WriteAllText(ownerInputPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                "Import-PostPublishCleanConsumerProofResult.ps1",
                "-OwnerInputPath",
                "artifacts/final-release/post-publish-clean-consumer-proof-result.ready.json",
                "-RequireExistingFiles",
                "-RequireHashMatch");
            RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict", "-FailOnNotProof");

            using JsonDocument importDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-import.json");
            JsonElement import = importDocument.RootElement;
            Assert.Equal("post-publish-clean-consumer-proof-result-import-ready", import.GetProperty("importState").GetString());
            Assert.True(import.GetProperty("proofCandidateReady").GetBoolean());
            Assert.Equal(0, import.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, import.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(import.GetProperty("sourceProofLinkageReady").GetBoolean());
            Assert.True(import.GetProperty("sourceGitHubActionsRunEvidenceReady").GetBoolean());
            Assert.True(import.GetProperty("sourceOwnerPublicPublishResultReady").GetBoolean());
            Assert.True(import.GetProperty("sourcePublicPackageDownloadProofReady").GetBoolean());
            Assert.Equal("123456789", import.GetProperty("sourceGitHubActionsRunId").GetString());
            Assert.Equal("https://api.nuget.org/v3/registration5-semver1/jyppx.tensorrt.csharp.api/index.json", import.GetProperty("sourceOwnerPublicPackageUrl").GetString());
            Assert.Equal(downloadedManagedPackageSha256, import.GetProperty("sourceOwnerPublicPackageSha256").GetString());
            Assert.False(import.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(import.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(import.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(import.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(import.GetProperty("canCloseReleaseIssue").GetBoolean());

            using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("post-publish-clean-consumer-proof-result-validation-ready", validation.GetProperty("validationState").GetString());
            Assert.True(validation.GetProperty("proofCandidateReady").GetBoolean());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(validation.GetProperty("sourceProofLinkageReady").GetBoolean());
            Assert.True(validation.GetProperty("sourceGitHubActionsRunEvidenceReady").GetBoolean());
            Assert.True(validation.GetProperty("sourceOwnerPublicPublishResultReady").GetBoolean());
            Assert.True(validation.GetProperty("sourcePublicPackageDownloadProofReady").GetBoolean());
            Assert.Equal(downloadedManagedPackageSha256, validation.GetProperty("downloadedManagedPackageSha256").GetString());
            Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
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

    [Fact]
    public void PostPublishProofResultFeedsConvergenceLaneAndCrossLaneChecksWithoutPromotion()
    {
        RunPowerShell("Import-PostPublishCleanConsumerProofResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict");
        RunPowerShell("Export-PostPublishCleanConsumerResultConvergence.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerResultConvergence.ps1", "-Strict");

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("post-publish-clean-consumer-result-convergence.json");
        JsonElement convergence = convergenceDocument.RootElement;
        string[] laneIds = convergence.GetProperty("lanes")
            .EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();

        Assert.Contains("post-publish-clean-consumer-proof-result", laneIds);
        Assert.True(convergence.GetProperty("crossLaneConsistencyChecks").GetArrayLength() >= 7);
        Assert.Equal(0, convergence.GetProperty("failedConsistencyBlockerCount").GetInt32());
        Assert.True(convergence.GetProperty("failedConsistencyActionRequiredCount").GetInt32() > 0);
        Assert.False(convergence.GetProperty("postPublishCleanConsumerProofResultReady").GetBoolean());
        Assert.False(convergence.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-clean-consumer-result-convergence-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-result-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("crossLaneConsistencyCheckCount").GetInt32() >= 7);
        Assert.Equal(0, validation.GetProperty("failedConsistencyBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedConsistencyActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void PostPublishProofResultRejectsForbiddenSubstitutesWithoutMakingValidationStructurallyInvalid()
    {
        Dictionary<string, object?> values = new()
        {
            ["recordKind"] = "post-publish-clean-consumer-proof-result-owner-input",
            ["ownerInputState"] = "owner-filled-post-publish-clean-consumer-proof-result",
            ["publicPackageSourceUrl"] = "file:///local/feed",
            ["publicPackageUrl"] = Path.Combine(RepositoryPaths.Root, "artifacts", "package-managed-dry-run", "package.nupkg"),
            ["publicPackageSourceKind"] = "local-feed",
            ["managedPackageId"] = "JYPPX.TensorRT.CSharp.API",
            ["managedPackageVersion"] = "4.0.0",
            ["runtimePackageId"] = "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22",
            ["runtimePackageVersion"] = "4.0.0",
            ["runtimePackageKey"] = "win-x64-trt11.0-cuda13.2-cudnn9.22",
            ["downloadedManagedPackagePath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "package-managed-dry-run", "managed.nupkg"),
            ["downloadedManagedPackageSha256"] = new string('a', 64),
            ["downloadedRuntimePackagePath"] = Path.Combine(RepositoryPaths.Root, "artifacts", "github-actions-runs", "runtime.nupkg"),
            ["downloadedRuntimePackageSha256"] = new string('b', 64),
            ["cleanConsumerRoot"] = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision"),
            ["consumerProjectPath"] = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "YoloVision.csproj"),
            ["restoreCommand"] = "dotnet restore --source ./artifacts/package-managed-dry-run",
            ["buildCommand"] = "dotnet build with ProjectReference",
            ["runCommand"] = "dotnet run direct .nupkg local feed",
            ["installLogPath"] = "install.log",
            ["installLogSha256"] = new string('c', 64),
            ["restoreLogPath"] = "restore.log",
            ["restoreLogSha256"] = new string('d', 64),
            ["buildLogPath"] = "build.log",
            ["buildLogSha256"] = new string('e', 64),
            ["runLogPath"] = "run.log",
            ["runLogSha256"] = new string('f', 64),
            ["smokeStdoutPath"] = "stdout.log",
            ["smokeStdoutSha256"] = new string('1', 64),
            ["smokeStderrPath"] = "stderr.log",
            ["smokeStderrSha256"] = new string('2', 64),
            ["nativeAssetListingPath"] = "native-assets.txt",
            ["nativeAssetListingSha256"] = new string('3', 64),
            ["dotnetInfoPath"] = "dotnet-info.txt",
            ["dotnetInfoSha256"] = new string('4', 64),
            ["exitCode"] = 0,
            ["confirmsPostPublish"] = true,
            ["confirmsNotPrePublishSmoke"] = true,
            ["hostMetadata"] = new Dictionary<string, object?>
            {
                ["os"] = "Windows 11",
                ["arch"] = "x64",
                ["rid"] = "win-x64",
                ["gpuName"] = "NVIDIA RTX",
                ["nvidiaDriver"] = "555.00",
                ["cudaRuntimeToolkit"] = "13.2",
                ["tensorrt"] = "11.0",
                ["cudnn"] = "9.22",
            },
            ["ownerReviewer"] = "owner",
            ["ownerReviewedAtUtc"] = DateTimeOffset.UtcNow.ToString("O"),
        };

        string ownerInputPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-clean-consumer-proof-result.misuse.json");
        File.WriteAllText(ownerInputPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

        RunPowerShell(
            "Import-PostPublishCleanConsumerProofResult.ps1",
            "-OwnerInputPath",
            "artifacts/final-release/post-publish-clean-consumer-proof-result.misuse.json");
        RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict");

        using JsonDocument importDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-import.json");
        JsonElement import = importDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-proof-result-required", import.GetProperty("importState").GetString());
        Assert.False(import.GetProperty("proofCandidateReady").GetBoolean());
        Assert.True(import.GetProperty("failedBlockerCount").GetInt32() > 0);
        Assert.Contains(import.GetProperty("findings").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "public-package-source-public-https" &&
            item.GetProperty("severity").GetString() == "blocker");
        Assert.Contains(import.GetProperty("findings").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "clean-consumer-root-outside-repository" &&
            item.GetProperty("severity").GetString() == "blocker");

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-result-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("post-publish-clean-consumer-proof-result-validation-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
    }

    private static void AssertCanonicalContractFields(JsonElement contract)
    {
        string[] names = contract.GetProperty("requiredFields")
            .EnumerateArray()
            .Select(static item => item.GetProperty("name").GetString()!)
            .ToArray();

        string[] required =
        [
            "cleanExternalConsumer.root",
            "cleanExternalConsumer.projectPath",
            "cleanExternalConsumer.restoreLogSha256",
            "cleanExternalConsumer.buildLogSha256",
            "cleanExternalConsumer.smokeLogSha256",
            "cleanExternalConsumer.stdoutLogSha256",
            "cleanExternalConsumer.stderrLogSha256",
            "hostMetadata.osDescription",
            "hostMetadata.gpuName",
            "hostMetadata.cudaDriverVersion",
            "hostMetadata.cudaRuntimeVersion",
            "hostMetadata.cudnnVersion",
            "hostMetadata.tensorRtVersion",
            "hostMetadata.tensorRtLine",
            "ownerReview.reviewer",
            "ownerReview.reviewedAtUtc",
            "ownerReview.approvalState",
            "forbiddenSubstituteCounts.projectReferenceCount",
            "forbiddenSubstituteCounts.localFeedReferenceCount",
            "forbiddenSubstituteCounts.directNupkgReferenceCount",
            "forbiddenSubstituteCounts.buildOnlyCount",
            "forbiddenSubstituteCounts.dependencyProbeOnlyCount",
            "forbiddenSubstituteCounts.blockedByDriverOnlyCount",
            "sourceProofs.githubActionsRunEvidenceReady",
            "sourceProofs.githubActionsRunId",
            "sourceProofs.githubActionsRunUrl",
            "sourceProofs.githubActionsHeadSha",
            "sourceProofs.ownerPublicPublishResultReady",
            "sourceProofs.publicPackageDownloadProofReady",
            "sourceProofs.publicPackageUrl",
            "sourceProofs.publicPackageVersion",
            "sourceProofs.publicPackageSha256",
            "sourceProofs.managedPackageDownloadUrl",
            "sourceProofs.runtimePackageDownloadUrl",
            "sourceProofs.githubReleaseAssetUrl",
            "sourceProofs.githubReleaseAssetSha256",
        ];

        foreach (string name in required)
        {
            Assert.Contains(name, names);
        }
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string AddEvidence(Dictionary<string, object?> values, string root, string pathField, string hashField, string fileName)
    {
        string path = Path.Combine(root, fileName);
        File.WriteAllText(path, $"post-publish evidence {fileName}");
        string hash = Sha256(path);
        values[pathField] = path;
        values[hashField] = hash;
        return hash;
    }

    private static void WriteReadySourceProofValidations(string publicPackageUrl, string publicPackageVersion, string publicPackageSha256)
    {
        string outputRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release");
        Directory.CreateDirectory(outputRoot);

        File.WriteAllText(
            Path.Combine(outputRoot, "github-actions-run-evidence-import-validation.json"),
            JsonSerializer.Serialize(
                new Dictionary<string, object?>
                {
                    ["recordKind"] = "github-actions-run-evidence-import-validation",
                    ["validationState"] = "github-actions-run-evidence-ready",
                    ["githubActionsRunEvidenceReady"] = true,
                    ["runId"] = "123456789",
                    ["runUrl"] = "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/123456789",
                    ["headSha"] = "0123456789abcdef0123456789abcdef01234567",
                    ["workflowRunLogSha256"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    ["artifactManifestSha256"] = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                },
                new JsonSerializerOptions { WriteIndented = true }));

        File.WriteAllText(
            Path.Combine(outputRoot, "owner-public-publish-execution-result-candidate-validation.json"),
            JsonSerializer.Serialize(
                new Dictionary<string, object?>
                {
                    ["recordKind"] = "owner-public-publish-execution-result-candidate-validation",
                    ["validationState"] = "owner-public-publish-execution-result-candidate-ready",
                    ["proofCandidateReady"] = true,
                    ["publicPackageUrl"] = publicPackageUrl,
                    ["publicPackageVersion"] = publicPackageVersion,
                    ["publicPackageSha256"] = publicPackageSha256,
                    ["githubReleaseAssetUrl"] = "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/download/v4.0.0/JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg",
                    ["githubReleaseAssetSha256"] = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                },
                new JsonSerializerOptions { WriteIndented = true }));

        File.WriteAllText(
            Path.Combine(outputRoot, "public-package-download-proof-candidate-validation.json"),
            JsonSerializer.Serialize(
                new Dictionary<string, object?>
                {
                    ["recordKind"] = "public-package-download-proof-candidate-validation",
                    ["validationState"] = "public-package-download-proof-candidate-ready",
                    ["proofCandidateReady"] = true,
                    ["publicPackageDownloadProofCandidateReady"] = true,
                    ["managedPackagePageUrl"] = publicPackageUrl,
                    ["managedPackageDownloadUrl"] = "https://www.nuget.org/api/v2/package/JYPPX.TensorRT.CSharp.API/4.0.0",
                    ["runtimePackagePageUrl"] = "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22/4.0.0",
                    ["runtimePackageDownloadUrl"] = "https://www.nuget.org/api/v2/package/JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22/4.0.0",
                    ["sourceOwnerPublicPackageUrl"] = publicPackageUrl,
                    ["sourceOwnerPublicPackageVersion"] = publicPackageVersion,
                    ["sourceOwnerPublicPackageSha256"] = publicPackageSha256,
                    ["githubReleaseAssetUrl"] = "https://github.com/guojin-yan/TensorRT-CSharp-API/releases/download/v4.0.0/JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg",
                    ["githubReleaseAssetSha256"] = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                },
                new JsonSerializerOptions { WriteIndented = true }));
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        ProcessResult result = RunPowerShellRaw(scriptName, arguments);
        Assert.True(result.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");
        return result.Stdout;
    }

    private static string RunPowerShellExpectFailure(string scriptName, params string[] arguments)
    {
        ProcessResult result = RunPowerShellRaw(scriptName, arguments);
        Assert.NotEqual(0, result.ExitCode);
        return result.Stdout + result.Stderr;
    }

    private static ProcessResult RunPowerShellRaw(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return new ProcessResult(process.ExitCode, stdout, stderr);
    }

    private sealed record ProcessResult(int ExitCode, string Stdout, string Stderr);
}
