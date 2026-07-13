using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerRealProofGapMatrixTests
{
    [Fact]
    public void GapMatrixStaysBlockedAndListsAllRealOwnerEvidenceGaps()
    {
        RunPowerShell("Export-FinalOwnerRealProofGapMatrix.ps1");
        RunPowerShell("Test-FinalOwnerRealProofGapMatrix.ps1", "-Strict");

        using JsonDocument matrixDocument = ReadFinalReleaseJson("final-owner-real-proof-gap-matrix.json");
        JsonElement matrix = matrixDocument.RootElement;

        Assert.Equal("final-owner-real-proof-gap-matrix", matrix.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-proof-gaps-remain", matrix.GetProperty("matrixState").GetString());
        Assert.True(matrix.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(matrix.GetProperty("passed").GetBoolean());
        Assert.True(matrix.GetProperty("gapCount").GetInt32() >= 10);
        AssertNonProof(matrix);

        string[] categories = matrix.GetProperty("gaps")
            .EnumerateArray()
            .Select(static gap => gap.GetProperty("category").GetString()!)
            .Distinct()
            .ToArray();

        foreach (string expected in new[]
        {
            "external-clean-consumer-runtime",
            "post-publish-clean-consumer-proof",
            "rollback-review",
            "final-close-decision",
            "release-evidence-refresh"
        })
        {
            Assert.Contains(expected, categories);
        }

        string[] gapTypes = matrix.GetProperty("gaps")
            .EnumerateArray()
            .Select(static gap => gap.GetProperty("gapType").GetString()!)
            .Distinct()
            .ToArray();

        foreach (string expected in new[]
        {
            "missing-field",
            "missing-file",
            "missing-sha256",
            "missing-host-metadata",
            "missing-owner-confirmation"
        })
        {
            Assert.Contains(expected, gapTypes);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-real-proof-gap-matrix-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-real-proof-gap-matrix-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("passed").GetBoolean());
        AssertNonProof(validation);
    }

    private static void AssertNonProof(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
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
