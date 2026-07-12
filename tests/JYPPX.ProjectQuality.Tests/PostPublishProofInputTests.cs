using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishProofInputTests
{
    [Fact]
    public void PostPublishProofTemplateExportsBlockedNonPublishingSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishProofInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishProofInput.ps1"), "-Strict");

        using JsonDocument templateDocument = ReadFinalReleaseJson("post-publish-proof-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("post-publish-proof-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-proof-required", template.GetProperty("validationState").GetString());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(template.GetProperty("postPublishProofReady").GetBoolean());
        Assert.False(template.GetProperty("isPostPublishProof").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-proof-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("post-publish-proof-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("postPublishProofReady").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
    }

    [Fact]
    public void PostPublishProofValidatorRejectsLocalConsumerSubstitutes()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), "trtsharp-postpublish-bad-" + Guid.NewGuid().ToString("N"));
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
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishProofInputTemplate.ps1"));
            using JsonDocument templateDocument = ReadFinalReleaseJson("post-publish-proof-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["cleanExternalConsumerRoot"] = fixtureRoot;
            values["consumerProjectPath"] = projectPath;
            values["publishedManagedPackageUrl"] = "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0";
            values["publishedRuntimePackageUrl"] = "https://nuget.pkg.github.com/guojin-yan/index.json";
            values["nugetPackageMetadataUrl"] = "https://api.nuget.org/v3/index.json";
            values["githubPackagesMetadataUrl"] = "https://nuget.pkg.github.com/guojin-yan/index.json";
            values["publishedManagedPackageSha256"] = SixtyFour("a");
            values["publishedRuntimePackageSha256"] = SixtyFour("b");
            values["downloadedManagedNupkgSha256"] = SixtyFour("c");
            values["downloadedRuntimeNupkgSha256"] = SixtyFour("d");
            values["exitCode"] = "0";
            values["nativeAssetsCopied"] = "true";
            values["dependencyProbeStatus"] = "dependency-probe-only";
            values["smokeStatus"] = "passed";

            string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-proof-input.misuse.json");
            File.WriteAllText(misusePath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishProofInput.ps1"),
                "-InputPath",
                "artifacts/final-release/post-publish-proof-input.misuse.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-proof-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("blocked-post-publish-proof-required", validation.GetProperty("validationState").GetString());
            Assert.False(validation.GetProperty("postPublishProofReady").GetBoolean());
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
    public void PostPublishProofValidatorAcceptsReadyShapeWithoutClosingReleaseIssue()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), "trtsharp-postpublish-good-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(fixtureRoot);
        string projectPath = Path.Combine(fixtureRoot, "GoodConsumer.csproj");
        string managedPackagePath = Path.Combine(fixtureRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
        string runtimePackagePath = Path.Combine(fixtureRoot, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg");
        string stdoutPath = Path.Combine(fixtureRoot, "stdout.log");
        string stderrPath = Path.Combine(fixtureRoot, "stderr.log");
        string reportPath = Path.Combine(fixtureRoot, "runtime-probe.json");

        try
        {
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
            CreateMinimalNupkg(managedPackagePath, "JYPPX.TensorRT.CSharp.API");
            CreateMinimalNupkg(runtimePackagePath, "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22");
            File.WriteAllText(stdoutPath, "post publish smoke passed");
            File.WriteAllText(stderrPath, string.Empty);
            File.WriteAllText(reportPath, """{"dependencyProbeStatus":"passed","smokeStatus":"passed"}""");

            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishProofInputTemplate.ps1"));
            using JsonDocument templateDocument = ReadFinalReleaseJson("post-publish-proof-input.template.json");
            Dictionary<string, object?> values = ToDictionary(templateDocument.RootElement);
            values["publishedAtUtc"] = DateTimeOffset.UtcNow.AddMinutes(-5).ToString("O");
            values["publishedManagedPackageUrl"] = "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0";
            values["publishedRuntimePackageUrl"] = "https://nuget.pkg.github.com/guojin-yan/index.json";
            values["publishedManagedPackageSha256"] = Sha256(managedPackagePath);
            values["publishedRuntimePackageSha256"] = Sha256(runtimePackagePath);
            values["nugetPackageMetadataUrl"] = "https://api.nuget.org/v3/index.json";
            values["githubPackagesMetadataUrl"] = "https://nuget.pkg.github.com/guojin-yan/index.json";
            values["downloadedManagedNupkgPath"] = managedPackagePath;
            values["downloadedManagedNupkgSha256"] = Sha256(managedPackagePath);
            values["downloadedRuntimeNupkgPath"] = runtimePackagePath;
            values["downloadedRuntimeNupkgSha256"] = Sha256(runtimePackagePath);
            values["cleanExternalConsumerRoot"] = fixtureRoot;
            values["consumerProjectPath"] = projectPath;
            values["restoreCommand"] = "dotnet restore GoodConsumer.csproj --source https://api.nuget.org/v3/index.json";
            values["buildCommand"] = "dotnet build GoodConsumer.csproj -c Release --no-restore";
            values["smokeCommand"] = "dotnet run --project GoodConsumer.csproj -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22";
            values["exitCode"] = "0";
            values["stdoutLogPath"] = stdoutPath;
            values["stdoutLogSha256"] = Sha256(stdoutPath);
            values["stderrLogPath"] = stderrPath;
            values["stderrLogSha256"] = Sha256(stderrPath);
            values["runtimeProbeReportPath"] = reportPath;
            values["runtimeProbeReportSha256"] = Sha256(reportPath);
            values["dependencyProbeStatus"] = "passed";
            values["smokeStatus"] = "passed";
            values["nativeAssetsCopied"] = "true";
            values["hostOs"] = "Windows";
            values["hostArchitecture"] = "x64";
            values["gpuName"] = "NVIDIA RTX";
            values["driverVersion"] = "580.00";
            values["cudaRuntimeVersion"] = "13.2";
            values["tensorRtVersion"] = "11.0";
            values["cudnnVersion"] = "9.22";
            values["ownerName"] = "Release Owner";
            values["ownerReviewedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");

            string readyPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-proof-input.ready.json");
            File.WriteAllText(readyPath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishProofInput.ps1"),
                "-InputPath",
                "artifacts/final-release/post-publish-proof-input.ready.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-proof-input-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("post-publish-proof-input-ready", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.True(validation.GetProperty("postPublishProofReady").GetBoolean());
            Assert.True(validation.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(validation.GetProperty("performsPublish").GetBoolean());
            Assert.False(validation.GetProperty("usesPublishToken").GetBoolean());
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
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void CreateMinimalNupkg(string path, string packageId)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        AddZipEntry(archive, "_rels/.rels", "<Relationships />");
        AddZipEntry(archive, $"{packageId}.nuspec", $"<package><metadata><id>{packageId}</id><version>4.0.0</version></metadata></package>");
        AddZipEntry(archive, "README.md", "# package");
    }

    private static void AddZipEntry(ZipArchive archive, string entryName, string content)
    {
        ZipArchiveEntry entry = archive.CreateEntry(entryName);
        using Stream stream = entry.Open();
        using StreamWriter writer = new(stream);
        writer.Write(content);
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string SixtyFour(string value)
    {
        return string.Concat(Enumerable.Repeat(value, 64));
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
