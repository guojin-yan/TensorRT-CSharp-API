using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionRealInputTemplateTests
{
    [Fact]
    public void RealInputTemplateMapsSkeletonFieldsWithoutPromotion()
    {
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Test-FinalOwnerExecutionInputSkeleton.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerExecutionRealInputTemplate.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputTemplate.ps1", "-Strict");

        using JsonDocument templateDocument = ReadFinalReleaseJson("final-owner-execution-real-input.template.json");
        JsonElement template = templateDocument.RootElement;

        Assert.Equal("final-owner-execution-real-input-template", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", template.GetProperty("templateState").GetString());
        Assert.True(template.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(template.GetProperty("readyForImport").GetBoolean());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(template.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(template.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(template.GetProperty("isReleaseCloseProof").GetBoolean());

        Assert.True(template.GetProperty("fieldValues").EnumerateObject().Count() >= 47);
        Assert.True(template.GetProperty("fileEvidence").EnumerateArray().Count() >= 8);
        Assert.True(template.GetProperty("hashEvidence").EnumerateArray().Count() >= 8);
        Assert.True(template.GetProperty("nonSubstituteConfirmations").EnumerateArray().Count() >= 10);

        foreach (JsonProperty field in template.GetProperty("fieldValues").EnumerateObject())
        {
            Assert.Equal("<owner-real-input-required>", field.Value.GetString());
        }

        foreach (string section in new[]
        {
            "owner",
            "fieldValues",
            "fileEvidence",
            "hashEvidence",
            "hostMetadata",
            "packageMetadata",
            "postPublishEvidence",
            "dualPackageRouteProof",
            "rollbackReview",
            "finalCloseDecision",
            "strictValidatorOutputs",
            "nonSubstituteConfirmations"
        })
        {
            Assert.True(template.TryGetProperty(section, out _), $"Missing template section: {section}");
        }

        string templateText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-execution-real-input.template.json"));
        foreach (string marker in new[]
        {
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "pre-publish smoke reused as post-publish proof",
            "nuget-small-bridge-core",
            "github-packages-full-runtime",
            "owner-dual-package-nuget-owner-authorization-url",
            "owner-dual-package-github-runtime-dll-resolution-report-path",
            "Test-DualPackagePublishPreflightMatrix.ps1",
            "Test-FinalCloseGateConvergence.ps1",
            "not runtime proof",
            "not post-publish proof",
            "not publish approval",
            "not release close approval",
            "not package push"
        })
        {
            Assert.Contains(marker, templateText, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument exampleDocument = ReadFinalReleaseJson("final-owner-execution-real-input.example.json");
        JsonElement example = exampleDocument.RootElement;
        Assert.Equal("final-owner-execution-real-input-example", example.GetProperty("recordKind").GetString());
        Assert.True(example.GetProperty("isExample").GetBoolean());
        Assert.False(example.GetProperty("readyForImport").GetBoolean());
        Assert.False(example.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-execution-real-input-template-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-real-input-template-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-template-ready", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("fieldValueCount").GetInt32() >= 47);
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
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

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
