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
