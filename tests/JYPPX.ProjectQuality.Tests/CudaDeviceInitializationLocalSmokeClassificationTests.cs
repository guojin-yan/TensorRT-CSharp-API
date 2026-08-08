using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaDeviceInitializationLocalSmokeClassificationTests
{
    [Fact]
    public void ExportAndValidationKeepCudaDeviceInitializationSmokeAsNonProof()
    {
        RunPowerShell("Export-CudaDeviceInitializationLocalSmokeClassification.ps1");
        string output = RunPowerShell("Test-CudaDeviceInitializationLocalSmokeClassification.ps1", "-Strict");

        Assert.Contains("ValidationState=cuda-device-initialization-local-smoke-classification-validation-passed-non-proof", output, StringComparison.Ordinal);

        using JsonDocument recordDocument = ReadFinalReleaseJson("cuda-device-initialization-local-smoke-classification.json");
        JsonElement record = recordDocument.RootElement;

        Assert.Equal("cuda-device-initialization-local-smoke-classification", record.GetProperty("recordKind").GetString());
        Assert.Equal("cuda-device-initialization-local-smoke-classified-non-proof", record.GetProperty("classificationState").GetString());
        Assert.Equal("local-smoke-not-external-proof", record.GetProperty("proofKind").GetString());
        Assert.Equal("smoke/CudaDeviceInitializationProofRunner/Program.cs", record.GetProperty("sourceSmokeRunner").GetString());
        Assert.True(record.GetProperty("preInitCallOrderReady").GetBoolean());
        Assert.True(record.GetProperty("skippedTrueIsForbiddenSubstitute").GetBoolean());
        AssertFalseProofFlags(record);

        string recordText = record.GetRawText();
        Assert.Contains("Skipped=True", recordText, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof", recordText, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", record.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("cuda-device-initialization-local-smoke-classification-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("cuda-device-initialization-local-smoke-classification-validation-passed-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("findingCount").GetInt32());
        AssertFalseProofFlags(validation);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static void AssertFalseProofFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        using Process process = new();
        process.StartInfo = new ProcessStartInfo
        {
            FileName = PowerShellHost.ResolveExecutable(),
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            WorkingDirectory = RepositoryPaths.Root
        };
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"PowerShell script failed ({scriptName}) with exit code {process.ExitCode}.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }

        return stdout + stderr;
    }
}
