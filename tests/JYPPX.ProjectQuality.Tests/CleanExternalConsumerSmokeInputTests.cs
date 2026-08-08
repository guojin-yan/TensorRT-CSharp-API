using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class CleanExternalConsumerSmokeInputTests
{
    [Fact]
    public void CleanExternalConsumerSmokeTemplateExportsBlockedNonPublishingInputSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerSmokeInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanExternalConsumerSmokeInput.ps1"), "-Strict");

        using JsonDocument templateDocument = ReadFinalReleaseJson("clean-external-consumer-smoke-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("clean-external-consumer-smoke-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-clean-external-consumer-smoke-required", template.GetProperty("validationState").GetString());
        Assert.Contains("--runtime-package-key", template.GetProperty("smokeCommand").GetString(), StringComparison.Ordinal);
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.Contains("ProjectReference", template.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("clean-external-consumer-smoke-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("clean-external-consumer-smoke-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-clean-external-consumer-smoke-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("cleanExternalConsumerSmokeReady").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string[] validationItemIds = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("clean-root-outside-repository", validationItemIds);
        Assert.Contains("consumer-project-outside-repository", validationItemIds);
        Assert.Contains("no-project-reference", validationItemIds);
        Assert.Contains("no-local-restore-source", validationItemIds);
        Assert.Contains("smoke-command-runtime-key", validationItemIds);
        Assert.Contains("dependency-probe-status-passed", validationItemIds);
    }

    [Fact]
    public void CleanExternalConsumerSmokeValidatorRejectsProjectReferenceSourcePathAndLocalFeed()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), "trtsharp-clean-consumer-bad-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(fixtureRoot);
        string projectPath = Path.Combine(fixtureRoot, "BadConsumer.csproj");
        File.WriteAllText(
            projectPath,
            $$"""
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <TargetFramework>net8.0</TargetFramework>
                <RestoreSources>{{RepositoryPaths.Root}}\artifacts\package-managed-dry-run</RestoreSources>
              </PropertyGroup>
              <ItemGroup>
                <ProjectReference Include="{{RepositoryPaths.Root}}\src\JYPPX.TensorRtSharp\JYPPX.TensorRtSharp.csproj" />
                <None Include="../src/not-allowed.nupkg" />
              </ItemGroup>
            </Project>
            """);

        try
        {
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerSmokeInputTemplate.ps1"));
            using JsonDocument templateDocument = ReadFinalReleaseJson("clean-external-consumer-smoke-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["cleanExternalConsumerRoot"] = fixtureRoot;
            values["consumerProjectPath"] = projectPath;
            values["exitCode"] = "0";
            values["startedAtUtc"] = DateTimeOffset.UtcNow.AddMinutes(-1).ToString("O");
            values["finishedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["nativeAssetsCopied"] = "true";
            values["dependencyProbeStatus"] = "dependency-probe-only";
            values["smokeStatus"] = "passed";

            string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "clean-external-consumer-smoke-input.misuse.json");
            File.WriteAllText(misusePath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanExternalConsumerSmokeInput.ps1"),
                "-InputPath",
                "artifacts/final-release/clean-external-consumer-smoke-input.misuse.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("clean-external-consumer-smoke-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("blocked-clean-external-consumer-smoke-required", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.False(validation.GetProperty("cleanExternalConsumerSmokeReady").GetBoolean());
            Assert.True(validation.GetProperty("consumerProjectUsesProjectReference").GetBoolean());
            Assert.True(validation.GetProperty("consumerProjectUsesLocalRestoreSource").GetBoolean());
            Assert.True(validation.GetProperty("consumerProjectUsesDirectNupkg").GetBoolean());
            AssertValidationItemFailed(validation, "no-project-reference");
            AssertValidationItemFailed(validation, "no-src-path-reference");
            AssertValidationItemFailed(validation, "no-repository-absolute-path");
            AssertValidationItemFailed(validation, "no-local-restore-source");
            AssertValidationItemFailed(validation, "no-direct-nupkg-reference");
            AssertValidationItemFailed(validation, "dependency-probe-status-passed");
        }
        finally
        {
            if (Directory.Exists(fixtureRoot))
            {
                Directory.Delete(fixtureRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void CleanExternalConsumerSmokeValidatorAcceptsExternalPackageReferenceSmokeShapeWithoutPromotingProofClaim()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), "trtsharp-clean-consumer-good-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(fixtureRoot);
        string projectPath = Path.Combine(fixtureRoot, "GoodConsumer.csproj");
        string stdoutPath = Path.Combine(fixtureRoot, "stdout.log");
        string stderrPath = Path.Combine(fixtureRoot, "stderr.log");
        string reportPath = Path.Combine(fixtureRoot, "runtime-probe.json");
        File.WriteAllText(
            projectPath,
            """
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <TargetFramework>net8.0</TargetFramework>
              </PropertyGroup>
              <ItemGroup>
                <PackageReference Include="JYPPX.TensorRT.CSharp.API" Version="4.0.0" />
              </ItemGroup>
            </Project>
            """);
        File.WriteAllText(stdoutPath, "runtime smoke passed");
        File.WriteAllText(stderrPath, string.Empty);
        File.WriteAllText(reportPath, """{"dependencyProbeStatus":"passed","smokeStatus":"passed"}""");

        try
        {
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerSmokeInputTemplate.ps1"));
            using JsonDocument templateDocument = ReadFinalReleaseJson("clean-external-consumer-smoke-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["cleanExternalConsumerRoot"] = fixtureRoot;
            values["consumerProjectPath"] = projectPath;
            values["restoreCommand"] = "dotnet restore GoodConsumer.csproj --source https://api.nuget.org/v3/index.json";
            values["buildCommand"] = "dotnet build GoodConsumer.csproj -c Release --no-restore";
            values["smokeCommand"] = "dotnet run --project GoodConsumer.csproj -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22";
            values["exitCode"] = "0";
            values["startedAtUtc"] = DateTimeOffset.UtcNow.AddMinutes(-1).ToString("O");
            values["finishedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");
            values["stdoutLogPath"] = stdoutPath;
            values["stdoutLogSha256"] = Sha256(stdoutPath);
            values["stderrLogPath"] = stderrPath;
            values["stderrLogSha256"] = Sha256(stderrPath);
            values["runtimeProbeReportPath"] = reportPath;
            values["runtimeProbeReportSha256"] = Sha256(reportPath);
            values["hostOs"] = "Windows";
            values["hostArchitecture"] = "x64";
            values["gpuName"] = "NVIDIA RTX";
            values["driverVersion"] = "580.00";
            values["cudaRuntimeVersion"] = "13.2";
            values["tensorRtVersion"] = "11.0";
            values["cudnnVersion"] = "9.22";
            values["nativeAssetsCopied"] = "true";
            values["dependencyProbeStatus"] = "passed";
            values["smokeStatus"] = "passed";

            string readyPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "clean-external-consumer-smoke-input.ready.json");
            File.WriteAllText(readyPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanExternalConsumerSmokeInput.ps1"),
                "-InputPath",
                "artifacts/final-release/clean-external-consumer-smoke-input.ready.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("clean-external-consumer-smoke-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("clean-external-consumer-smoke-input-ready", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(validation.GetProperty("cleanExternalConsumerSmokeReady").GetBoolean());
            Assert.False(validation.GetProperty("canClaimPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(fixtureRoot))
            {
                Directory.Delete(fixtureRoot, recursive: true);
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
            static property => property.Value.ValueKind switch
            {
                JsonValueKind.True => (object?)true,
                JsonValueKind.False => false,
                JsonValueKind.Array => property.Value.EnumerateArray().Select(static item => item.GetString()).ToArray(),
                _ => property.Value.GetString()
            });
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream)).ToLowerInvariant();
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
