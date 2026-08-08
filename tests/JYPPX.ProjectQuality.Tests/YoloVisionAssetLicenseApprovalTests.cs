using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionAssetLicenseApprovalTests
{
    [Fact]
    public void TemplateAndValidatorKeepHashReadyAssetsBehindOwnerLicenseApproval()
    {
        string exportOutput = RunPowerShell(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Export-YoloVisionAssetLicenseApprovalTemplate.ps1"));
        Assert.Contains("ApprovalState=template-only-owner-action-required", exportOutput, StringComparison.Ordinal);

        string validationOutput = RunPowerShell(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-YoloVisionAssetLicenseApprovalRecord.ps1"));
        Assert.Contains("ValidationState=owner-action-required", validationOutput, StringComparison.Ordinal);

        string templatePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "reference-assets",
            "asset-license-approval-template.json");
        using JsonDocument templateDocument = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement template = templateDocument.RootElement;

        Assert.Equal("yolovision-reference-asset-license-approval", template.GetProperty("recordKind").GetString());
        Assert.Equal("YoloVision", template.GetProperty("sampleName").GetString());
        Assert.Equal("template-only-owner-action-required", template.GetProperty("approvalState").GetString());
        Assert.False(template.GetProperty("redistributionAllowed").GetBoolean());
        Assert.False(template.GetProperty("publicRepositoryAllowed").GetBoolean());
        Assert.False(template.GetProperty("allAssetsApproved").GetBoolean());
        Assert.True(template.GetProperty("allHashesMatchAcquisitionReport").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());

        JsonElement[] assets = template.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(new[] { "model", "labels", "input-image" }, assets.Select(static asset => asset.GetProperty("role").GetString()).ToArray());
        Assert.All(assets, static asset =>
        {
            Assert.True(asset.GetProperty("fileReady").GetBoolean());
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
            Assert.Equal(asset.GetProperty("expectedSha256").GetString(), asset.GetProperty("actualSha256").GetString());
            Assert.Equal("owner-required", asset.GetProperty("ownerLicenseName").GetString());
            Assert.Equal("owner-required", asset.GetProperty("ownerLicenseUri").GetString());
            Assert.False(asset.GetProperty("ownerRedistributionApproved").GetBoolean());
            Assert.False(asset.GetProperty("ownerPublicRepositoryApproved").GetBoolean());
        });

        string serializedTemplate = template.GetRawText();
        Assert.Contains("TensorRT SLA alone", serializedTemplate, StringComparison.Ordinal);
        Assert.Contains("SHA256 match alone", serializedTemplate, StringComparison.Ordinal);
        Assert.Contains("local installation file presence", serializedTemplate, StringComparison.Ordinal);

        string validationPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "reference-assets",
            "asset-license-approval-validation.json");
        using JsonDocument validationDocument = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("yolovision-reference-asset-license-approval-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("ownerActionRequiredCount").GetInt32() > 0);
        Assert.True(validation.GetProperty("allHashesMatchAcquisitionReport").GetBoolean());
        Assert.False(validation.GetProperty("allAssetsApproved").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "reference-assets",
            "asset-license-approval-validation.md"));
        Assert.Contains("TensorRT SLA text cannot substitute", markdown, StringComparison.Ordinal);
        Assert.Contains("ownerLicenseUri", markdown, StringComparison.Ordinal);
        Assert.Contains("ownerSignature", markdown, StringComparison.Ordinal);
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(180_000), $"Command timed out: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        Assert.Equal(0, process.ExitCode);
        Assert.True(string.IsNullOrWhiteSpace(stderr), stderr);
        return stdout;
    }
}
