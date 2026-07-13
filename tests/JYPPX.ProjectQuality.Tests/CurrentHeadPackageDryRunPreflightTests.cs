using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CurrentHeadPackageDryRunPreflightTests
{
    [Fact]
    public void PreflightClassifiesOldPackageDryRunAsOwnerAuthorizationRequired()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-current-head-dry-run-preflight-" + Guid.NewGuid().ToString("N"));
        string currentHead = "892a6d525741a93ec8dd3ba009071c626c3c90ee";
        string oldHead = "72d65909a120e1e550568e01796bbbdb5ec2b2e4";
        string sourceEvidencePath = Path.Combine(tempRoot, "source-quality.json");
        string packageDryRunEvidencePath = Path.Combine(tempRoot, "package-dry-run.json");
        string outputPath = Path.Combine(tempRoot, "current-head-package-dry-run-preflight.json");
        string markdownPath = Path.Combine(tempRoot, "current-head-package-dry-run-preflight.md");

        try
        {
            Directory.CreateDirectory(tempRoot);
            File.WriteAllText(
                sourceEvidencePath,
                $$"""
                {
                  "recordKind": "github-actions-run-evidence-import",
                  "evidenceState": "source-quality-run-evidence-ready",
                  "importMode": "source-quality-only",
                  "runId": "29230784443",
                  "runUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29230784443",
                  "headSha": "{{currentHead}}",
                  "canClaimGitHubActionsSourceQualityForRun": true,
                  "canClaimGitHubActionsPackageDryRunPackForRun": false,
                  "performsPublish": false,
                  "canPublishPublicly": false,
                  "canCloseReleaseIssue": false
                }
                """);
            File.WriteAllText(
                packageDryRunEvidencePath,
                $$"""
                {
                  "recordKind": "github-actions-run-evidence-import",
                  "importMode": "package-dry-run",
                  "runId": "29160655818",
                  "runUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29160655818",
                  "headSha": "{{oldHead}}",
                  "canClaimGitHubActionsSourceQualityForRun": true,
                  "canClaimGitHubActionsPackageDryRunPackForRun": true,
                  "canClaimNuGetPublished": false,
                  "canClaimGitHubPackagesPublished": false,
                  "performsPublish": false,
                  "canPublishPublicly": false,
                  "canCloseReleaseIssue": false,
                  "nupkgPackages": [
                    { "fileName": "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg", "sha256": "7c3d590b233f4b480c75bac073778160b8b60d87b16acf70cb86e6bcbace0247" }
                  ]
                }
                """);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-CurrentHeadPackageDryRunPreflight.ps1"),
                "-SourceQualityRunEvidenceImportPath",
                sourceEvidencePath,
                "-PackageDryRunEvidenceImportPath",
                packageDryRunEvidencePath,
                "-CurrentHead",
                currentHead,
                "-OutputPath",
                outputPath,
                "-MarkdownOutputPath",
                markdownPath);

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
            JsonElement root = document.RootElement;

            Assert.Equal("current-head-package-dry-run-preflight", root.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-owner-authorization-required", root.GetProperty("state").GetString());
            Assert.Equal(currentHead, root.GetProperty("currentHead").GetString());
            Assert.True(root.GetProperty("sourceQualityRunEvidencePresent").GetBoolean());
            Assert.True(root.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
            Assert.True(root.GetProperty("sourceQualityHeadMatchesCurrentHead").GetBoolean());
            Assert.True(root.GetProperty("packageDryRunEvidencePresent").GetBoolean());
            Assert.True(root.GetProperty("packageDryRunCanClaimPackForRun").GetBoolean());
            Assert.False(root.GetProperty("packageDryRunHeadMatchesCurrentHead").GetBoolean());
            Assert.False(root.GetProperty("canClaimGitHubActionsPackageDryRunPackForCurrentHead").GetBoolean());
            Assert.True(root.GetProperty("packageDryRunRequiresOwnerAuthorization").GetBoolean());
            Assert.True(root.GetProperty("manualWorkflowDispatchNotPerformed").GetBoolean());
            Assert.False(root.GetProperty("performsPublish").GetBoolean());
            Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

            JsonElement currentHeadItem = root.GetProperty("checks")
                .EnumerateArray()
                .Single(static item => item.GetProperty("id").GetString() == "package-dry-run-current-head-claim-ready");
            Assert.False(currentHeadItem.GetProperty("passed").GetBoolean());

            string markdown = File.ReadAllText(markdownPath);
            Assert.Contains("blocked-owner-authorization-required", markdown, StringComparison.Ordinal);
            Assert.Contains("manualWorkflowDispatchNotPerformed", markdown, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static void RunPowerShell(string scriptPath, params string[] arguments)
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
    }
}
