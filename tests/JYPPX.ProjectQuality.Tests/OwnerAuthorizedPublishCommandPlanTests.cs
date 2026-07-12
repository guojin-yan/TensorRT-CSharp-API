using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerAuthorizedPublishCommandPlanTests
{
    [Fact]
    public void OwnerAuthorizedPublishCommandPlanRemainsPlaceholderOnlyAndNonPublishing()
    {
        string exportScript = Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerAuthorizedPublishCommandPlan.ps1");
        string validationScript = Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerAuthorizedPublishCommandPlan.ps1");
        Assert.True(File.Exists(exportScript), "Owner authorized publish command plan export script must exist.");
        Assert.True(File.Exists(validationScript), "Owner authorized publish command plan validator must exist.");

        string scriptText = File.ReadAllText(exportScript);
        Assert.Contains("dotnet nuget push", scriptText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("placeholderOnly = $true", scriptText, StringComparison.Ordinal);
        Assert.Contains("materializedExecutableCommand = \"\"", scriptText, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", scriptText, StringComparison.Ordinal);
        Assert.Contains("modelExecutionForbidden", scriptText, StringComparison.Ordinal);
        Assert.Contains("Post-publish verification can close the release issue only after real channel publication", scriptText, StringComparison.Ordinal);

        RunPowerShell(exportScript);
        RunPowerShell(validationScript);

        using JsonDocument planDocument = ReadFinalReleaseJson("owner-authorized-publish-command-plan.json");
        JsonElement plan = planDocument.RootElement;
        Assert.Equal("owner-authorized-publish-command-plan", plan.GetProperty("recordKind").GetString());
        Assert.False(plan.GetProperty("performsPublish").GetBoolean());
        Assert.True(plan.GetProperty("requiresExplicitOwnerAuthorization").GetBoolean());
        Assert.False(plan.GetProperty("canMaterializeExecutableCommands").GetBoolean());
        Assert.Contains("blocked", plan.GetProperty("planState").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(plan.GetProperty("realPostPublishVerificationReady").GetBoolean());

        JsonElement[] commands = plan.GetProperty("publishCommands").EnumerateArray().ToArray();
        Assert.NotEmpty(commands);
        Assert.Contains(commands, command => command.GetProperty("command").GetString()!.Contains("dotnet nuget push", StringComparison.OrdinalIgnoreCase));
        foreach (JsonElement command in commands)
        {
            Assert.True(command.GetProperty("placeholderOnly").GetBoolean());
            Assert.False(command.GetProperty("authorized").GetBoolean());
            Assert.False(command.GetProperty("executable").GetBoolean());
            Assert.False(command.GetProperty("performsPublish").GetBoolean());
            Assert.Equal(string.Empty, command.GetProperty("materializedExecutableCommand").GetString());
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-authorized-publish-command-plan-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-authorized-publish-command-plan-validation", validation.GetProperty("validationKind").GetString());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canMaterializeExecutableCommands").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedValidationItemCount").GetInt32());
        Assert.True(validation.GetProperty("publishCommandsPlaceholderOnly").GetBoolean());
        Assert.False(validation.GetProperty("realPostPublishVerificationReady").GetBoolean());
        Assert.Contains("blocked", validation.GetProperty("postPublishVerificationCollectionPackageState").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.True(validation.GetProperty("postPublishVerificationCollectionPackageStepCount").GetInt32() > 0);
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
