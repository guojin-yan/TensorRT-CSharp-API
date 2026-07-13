using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class MultiVersionOnnxRuntimeEvidenceMatrixTests
{
    [Fact]
    public void ExportedMatrixKeepsVersionEvidenceAndProofBoundariesExplicit()
    {
        string output = RunPowerShell(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Export-MultiVersionOnnxRuntimeEvidenceMatrix.ps1"));

        Assert.Contains("Multi-version runtime evidence matrix written.", output, StringComparison.Ordinal);
        Assert.Contains("Cases=19 Passed=15 Blocked=4", output, StringComparison.Ordinal);
        Assert.Contains("Synthetic=4 RealModel=11 PackageConsumer=0", output, StringComparison.Ordinal);

        string directory = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "real-case",
            "multi-version-onnx-runtime");
        string jsonPath = Path.Combine(directory, "multi-version-runtime-evidence-matrix.json");
        string markdownPath = Path.Combine(directory, "multi-version-runtime-evidence-matrix.md");

        Assert.True(File.Exists(jsonPath), $"Expected {jsonPath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;

        Assert.Equal("multi-version-onnx-runtime-evidence-matrix", root.GetProperty("recordKind").GetString());
        Assert.Equal("partially-verified", root.GetProperty("matrixState").GetString());
        Assert.Equal(19, root.GetProperty("caseCount").GetInt32());
        Assert.Equal(15, root.GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(4, root.GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal(4, root.GetProperty("syntheticRuntimeCaseCount").GetInt32());
        Assert.Equal(11, root.GetProperty("realModelRuntimeCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("packageConsumerRuntimeCaseCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement[] cases = root.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(19, cases.Length);
        Assert.All(cases, static item =>
        {
            Assert.False(item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("proofBoundary").GetString()));

            JsonElement bridge = Assert.Single(
                item.GetProperty("evidence").EnumerateArray(),
                static artifact => artifact.GetProperty("kind").GetString() == "native-bridge");
            Assert.True(bridge.GetProperty("exists").GetBoolean());
            Assert.Equal(64, bridge.GetProperty("sha256").GetString()!.Length);

            foreach (JsonElement artifact in item.GetProperty("evidence").EnumerateArray())
            {
                if (artifact.GetProperty("exists").GetBoolean())
                {
                    Assert.Equal(64, artifact.GetProperty("sha256").GetString()!.Length);
                    Assert.True(artifact.GetProperty("lengthBytes").GetInt64() > 0);
                }
            }
        });

        JsonElement trt8Cuda11 = FindCase(cases, "trt8-cuda11-identity");
        Assert.Equal("passed", trt8Cuda11.GetProperty("state").GetString());
        Assert.Equal("synthetic-input-runtime", trt8Cuda11.GetProperty("proofClassification").GetString());
        Assert.True(trt8Cuda11.GetProperty("inferenceRan").GetBoolean());
        Assert.True(trt8Cuda11.GetProperty("outputMatch").GetBoolean());

        JsonElement trt10Cuda11Identity = FindCase(cases, "trt10-cuda11-identity");
        Assert.Equal("10.11.0", trt10Cuda11Identity.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("11.8", trt10Cuda11Identity.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal("synthetic-input-runtime", trt10Cuda11Identity.GetProperty("proofClassification").GetString());
        Assert.True(trt10Cuda11Identity.GetProperty("success").GetBoolean());
        Assert.True(trt10Cuda11Identity.GetProperty("outputMatch").GetBoolean());

        JsonElement trt10Cuda11Mnist = FindCase(cases, "trt10-cuda11-mnist-digit-7");
        Assert.Equal("real-model-runtime", trt10Cuda11Mnist.GetProperty("proofClassification").GetString());
        Assert.Equal(7, trt10Cuda11Mnist.GetProperty("result").GetProperty("expectedDigit").GetInt32());
        Assert.Equal(7, trt10Cuda11Mnist.GetProperty("result").GetProperty("predictedDigit").GetInt32());
        Assert.True(trt10Cuda11Mnist.GetProperty("result").GetProperty("confidence").GetDouble() >= 0.9);
        Assert.Equal("Input3", trt10Cuda11Mnist.GetProperty("tensorMetadata").GetProperty("inputName").GetString());
        Assert.Equal("Plus214_Output_0", trt10Cuda11Mnist.GetProperty("tensorMetadata").GetProperty("outputName").GetString());

        JsonElement[] cuda12MnistCases = cases
            .Where(static item => item.GetProperty("id").GetString()!.StartsWith("trt10-cuda12-mnist-digit-", StringComparison.Ordinal))
            .OrderBy(static item => item.GetProperty("result").GetProperty("expectedDigit").GetInt32())
            .ToArray();
        Assert.Equal(10, cuda12MnistCases.Length);
        for (int digit = 0; digit <= 9; digit++)
        {
            JsonElement item = cuda12MnistCases[digit];
            Assert.Equal("passed", item.GetProperty("state").GetString());
            Assert.Equal("real-model-runtime", item.GetProperty("proofClassification").GetString());
            Assert.Equal(digit, item.GetProperty("result").GetProperty("expectedDigit").GetInt32());
            Assert.Equal(digit, item.GetProperty("result").GetProperty("predictedDigit").GetInt32());
            Assert.True(item.GetProperty("result").GetProperty("confidence").GetDouble() >= 0.9);
            Assert.True(item.GetProperty("outputMatch").GetBoolean());
        }

        string[] expectedBlockedIds =
        {
            "trt8-cuda11-mnist-digit-7",
            "trt8-cuda12-mnist-digit-7",
            "trt11-cuda12-identity",
            "trt11-cuda12-mnist-digit-7",
        };
        Assert.Equal(
            expectedBlockedIds.Order(StringComparer.Ordinal).ToArray(),
            root.GetProperty("blockedCaseIds").EnumerateArray()
                .Select(static item => item.GetString()!)
                .Order(StringComparer.Ordinal)
                .ToArray());

        foreach (string id in expectedBlockedIds.Take(2))
        {
            JsonElement blocked = FindCase(cases, id);
            Assert.Equal("blocked-by-cudnn8-runtime-missing", blocked.GetProperty("state").GetString());
            Assert.Contains(
                blocked.GetProperty("missingRuntimeAssets").EnumerateArray().Select(static item => item.GetString()!),
                static path => path.EndsWith("cudnn64_8.dll", StringComparison.OrdinalIgnoreCase));
        }

        foreach (string id in expectedBlockedIds.Skip(2))
        {
            JsonElement blocked = FindCase(cases, id);
            Assert.Equal("blocked-by-runtime-assets-missing", blocked.GetProperty("state").GetString());
            string[] missing = blocked.GetProperty("missingRuntimeAssets").EnumerateArray()
                .Select(static item => item.GetString()!)
                .ToArray();
            Assert.Contains(missing, static path => path.EndsWith("nvinfer_11.dll", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(missing, static path => path.EndsWith("nvinfer_plugin_11.dll", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(missing, static path => path.EndsWith("nvonnxparser_11.dll", StringComparison.OrdinalIgnoreCase));
        }

        string markdown = File.ReadAllText(markdownPath);
        Assert.Contains("案例总数：`19`", markdown, StringComparison.Ordinal);
        Assert.Contains("通过：`15`", markdown, StringComparison.Ordinal);
        Assert.Contains("环境阻塞：`4`", markdown, StringComparison.Ordinal);
        Assert.Contains("trt10-cuda11-mnist-digit-7", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-by-runtime-assets-missing", markdown, StringComparison.Ordinal);
        Assert.Contains("package-consumer runtime：`0`", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("canPublishPublicly：`True`", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("canCloseReleaseIssue：`True`", markdown, StringComparison.Ordinal);
    }

    private static JsonElement FindCase(IEnumerable<JsonElement> cases, string id)
    {
        return Assert.Single(cases, item => item.GetProperty("id").GetString() == id);
    }

    private static string RunPowerShell(string scriptPath)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start PowerShell.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(120_000), $"PowerShell timed out.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        Assert.Equal(0, process.ExitCode);
        Assert.True(string.IsNullOrWhiteSpace(stderr), stderr);
        return stdout;
    }
}
