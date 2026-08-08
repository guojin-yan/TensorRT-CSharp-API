using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerRuntimeProofOwnerInputSchemaTests
{
    [Fact]
    public void OwnerInputSchemaDocumentsFieldsValidatorsAndForbiddenSubstitutesWithoutPromotingProof()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputSchema.ps1");
        Assert.True(File.Exists(scriptPath), "Owner input schema export script must exist.");

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("package-consumer-runtime-proof-owner-input.schema.json", script, StringComparison.Ordinal);
        Assert.Contains("placeholderAllowed", script, StringComparison.Ordinal);
        Assert.Contains("validatorItemId", script, StringComparison.Ordinal);
        Assert.Contains("proofRole", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);

        RunPowerShell(scriptPath);

        using JsonDocument document = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.schema.json");
        JsonElement schema = document.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input-schema", schema.GetProperty("recordKind").GetString());
        Assert.Equal("package-consumer-runtime", schema.GetProperty("proofLineId").GetString());
        Assert.True(schema.GetProperty("fieldCount").GetInt32() >= 40);
        Assert.True(schema.GetProperty("placeholderAllowedFieldCount").GetInt32() >= 10);
        Assert.False(schema.GetProperty("performsPublish").GetBoolean());
        Assert.False(schema.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(schema.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(schema.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(schema.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] requiredFields =
        [
            "cleanExternalConsumerRoot",
            "consumerProjectPath",
            "publicPackageSource",
            "managedNupkgSha256",
            "runtimePackageKey",
            "runtimeNupkgSha256",
            "smokeCommand",
            "exitCode",
            "dependencyProbeStatus",
            "smokeStatus",
            "nativeAssetsCopied",
            "smokeLogSha256",
            "canPromoteProof"
        ];

        string[] dryRunContextFields =
        [
            "currentHead",
            "currentHeadPackageDryRunPreflightPath",
            "currentHeadPackageDryRunPreflightPresent",
            "currentHeadPackageDryRunPreflightState",
            "currentHeadPackageDryRunReady",
            "currentHeadPackageDryRunOwnerAuthorizationRequired",
            "currentHeadPackageDryRunBlockedReason",
            "currentHeadPackageDryRunOwnerAction",
            "sourceQualityRunEvidenceImportPath",
            "sourceQualityRunId",
            "sourceQualityRunUrl",
            "sourceQualityHeadSha",
            "sourceQualityRunEvidenceReady",
            "sourceGitHubActionsRunEvidenceImportPath",
            "packageDryRunEvidenceImportPath",
            "packageDryRunRunId",
            "packageDryRunRunUrl",
            "packageDryRunHeadSha",
            "packageDryRunHeadMatchesCurrentHead",
            "sourceGitHubActionsRunId",
            "sourceGitHubActionsRunUrl",
            "sourceHeadSha",
            "packageDryRunArtifactPath",
            "packageDryRunManagedNupkgSha256",
            "packageDryRunCanClaimPack",
            "packageDryRunCanClaimCurrentHeadPack",
            "packageDryRunRequiresOwnerAuthorization",
            "manualWorkflowDispatchNotPerformed",
            "isDryRunOnly",
            "isPublishedPackageProof",
            "isPackageConsumerRuntimeProof"
        ];

        JsonElement[] fields = schema.GetProperty("fields").EnumerateArray().ToArray();
        foreach (string fieldName in requiredFields)
        {
            JsonElement field = Assert.Single(fields, item => item.GetProperty("name").GetString() == fieldName);
            Assert.True(field.GetProperty("required").GetBoolean());
            Assert.False(field.GetProperty("placeholderAllowed").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(field.GetProperty("validatorItemId").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(field.GetProperty("proofRole").GetString()));
        }

        foreach (string fieldName in dryRunContextFields)
        {
            JsonElement field = Assert.Single(fields, item => item.GetProperty("name").GetString() == fieldName);
            Assert.False(field.GetProperty("required").GetBoolean());
            Assert.True(field.GetProperty("placeholderAllowed").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(field.GetProperty("validatorItemId").GetString()));
            Assert.Contains("not proof", field.GetProperty("proofRole").GetString()!, StringComparison.OrdinalIgnoreCase);
        }

        string[] forbiddenSubstitutes =
        [
            "local-feed",
            "project-reference",
            "direct-nupkg",
            "repository-path-leakage",
            "build-only",
            "dry-run",
            "github-actions-dry-run-nupkg",
            "dashboard",
            "template-placeholder",
            "gui-screenshot",
            "tensorrtexec-build-report-only"
        ];

        string[] actualSubstitutes = schema.GetProperty("forbiddenSubstitutes")
            .EnumerateArray()
            .Select(item => item.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string forbiddenSubstitute in forbiddenSubstitutes)
        {
            Assert.Contains(forbiddenSubstitute, actualSubstitutes);
        }

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-runtime-proof-owner-input.schema.md"));
        Assert.Contains("Package Consumer Runtime Proof Owner Input Schema", markdown, StringComparison.Ordinal);
        Assert.Contains("Forbidden Substitutes", markdown, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec build report", markdown, StringComparison.Ordinal);
        Assert.Contains("sourceQualityRunEvidenceImportPath", markdown, StringComparison.Ordinal);
        Assert.Contains("currentHeadPackageDryRunPreflightState", markdown, StringComparison.Ordinal);
        Assert.Contains("currentHeadPackageDryRunOwnerAction", markdown, StringComparison.Ordinal);
        Assert.Contains("packageDryRunEvidenceImportPath", markdown, StringComparison.Ordinal);
        Assert.Contains("packageDryRunCanClaimCurrentHeadPack", markdown, StringComparison.Ordinal);
        Assert.Contains("source-quality run 不能填充 dry-run package artifact", markdown, StringComparison.Ordinal);
        Assert.Contains("packageDryRunManagedNupkgSha256", markdown, StringComparison.Ordinal);
        Assert.Contains("不是 Owner 从 public feed 下载的 package hash", markdown, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof", markdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
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
