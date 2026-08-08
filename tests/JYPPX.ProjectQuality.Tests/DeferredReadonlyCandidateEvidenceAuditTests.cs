using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredReadonlyCandidateEvidenceAuditTests
{
    [Fact]
    public void CandidateEvidenceAuditClosesRepositoryLinkageWithoutRuntimePromotion()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadonlyCandidateEvidenceAudit.ps1"));
        Assert.Contains("Deferred readonly candidate evidence audit written", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-evidence-audit.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-evidence-audit.md");
        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;
        Assert.Equal("deferred-readonly-candidate-evidence-audit.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("deferred-readonly-candidate-evidence-audit", root.GetProperty("auditKind").GetString());
        Assert.Equal("eng/deferred-readonly-candidate-evidence-map.json", root.GetProperty("sourceEvidenceMap").GetString());
        Assert.Equal(16, root.GetProperty("candidateCount").GetInt32());
        Assert.Equal(8, root.GetProperty("implementationCandidateCount").GetInt32());
        int evidencePathCheckCount = root.GetProperty("evidencePathCheckCount").GetInt32();
        Assert.True(evidencePathCheckCount >= 242);
        Assert.Equal(34, root.GetProperty("manifestCheckCount").GetInt32());
        Assert.Equal(94, root.GetProperty("publicSurfaceCheckCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingEvidencePathCount").GetInt32());
        Assert.Equal(0, root.GetProperty("manifestFindingCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingPublicSurfaceCount").GetInt32());
        Assert.Equal(0, root.GetProperty("forbiddenPublicHandleCount").GetInt32());
        Assert.Equal(0, root.GetProperty("findingCount").GetInt32());
        Assert.True(root.GetProperty("allEvidencePathsExist").GetBoolean());
        Assert.True(root.GetProperty("allManifestRecordsValid").GetBoolean());
        Assert.True(root.GetProperty("allManagedPublicSurfacesPresent").GetBoolean());
        Assert.True(root.GetProperty("noForbiddenPublicHandles").GetBoolean());

        foreach (string flag in new[]
        {
            "isRuntimeExecutionProof",
            "isPackageConsumerRuntimeProof",
            "canPromoteRuntimeProof",
            "canPromoteReleaseProof",
            "canPublishPublicly",
            "canCloseReleaseIssue",
            "performsPublish",
        })
        {
            Assert.False(root.GetProperty(flag).GetBoolean(), flag);
        }

        JsonElement[] candidates = root.GetProperty("candidates").EnumerateArray().ToArray();
        Assert.Equal(16, candidates.Length);
        Assert.All(candidates, static candidate =>
        {
            Assert.Equal(0, candidate.GetProperty("candidateFindingCount").GetInt32());
            Assert.Equal(0, candidate.GetProperty("missingPathCount").GetInt32());
            Assert.Equal(0, candidate.GetProperty("missingPublicSurfaceCount").GetInt32());
            Assert.Equal(0, candidate.GetProperty("forbiddenPublicHandleCount").GetInt32());
            Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(candidate.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(candidate.GetProperty("canDeleteDeferredRecord").GetBoolean());
        });

        foreach (string candidateId in new[]
        {
            "plugin-field-metadata-001",
            "engine-layer-metadata-001",
            "engine-tensor-binding-002",
            "builder-config-readback-001",
            "error-recorder-snapshot-001",
        })
        {
            JsonElement candidate = candidates.Single(item => item.GetProperty("candidateId").GetString() == candidateId);
            Assert.Contains("8", candidate.GetProperty("manifestVersionLines").EnumerateArray().Select(item => item.GetString()));
            Assert.Contains("10", candidate.GetProperty("manifestVersionLines").EnumerateArray().Select(item => item.GetString()));
            Assert.Contains("11", candidate.GetProperty("manifestVersionLines").EnumerateArray().Select(item => item.GetString()));
            Assert.True(candidate.GetProperty("nativeEntryPointCount").GetInt32() > 0);
            Assert.True(candidate.GetProperty("manifestCount").GetInt32() > 0);
        }

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "deferred-readonly-candidate-evidence-audit.v1",
            evidencePathCheckCount.ToString(),
            "34",
            "94",
            "not a native runtime",
            "canPromoteRuntimeProof",
            "No findings.",
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void CandidateEvidenceAuditExportIsDeterministicAndKeepsPluginMacroLinkage()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadonlyCandidateEvidenceAudit.ps1");
        string evidenceMapPath = Path.Combine(RepositoryPaths.Root, "eng", "deferred-readonly-candidate-evidence-map.json");
        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-evidence-audit.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-evidence-audit.md");

        RunPowerShell(scriptPath);
        string firstJsonHash = HashFile(jsonPath);
        string firstMarkdownHash = HashFile(markdownPath);
        RunPowerShell(scriptPath);

        Assert.Equal(firstJsonHash, HashFile(jsonPath));
        Assert.Equal(firstMarkdownHash, HashFile(markdownPath));

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN", script, StringComparison.Ordinal);
        Assert.Contains("prefix-macro", File.ReadAllText(jsonPath), StringComparison.Ordinal);
        Assert.Contains("native/manifests/tensorrt/v10/trt10-twenty-sixth-batch-plugin-registry-inventory.manifest.json", File.ReadAllText(jsonPath), StringComparison.Ordinal);
        Assert.Contains("native/manifests/tensorrt/v11/trt11-twenty-sixth-batch-plugin-registry-inventory.manifest.json", File.ReadAllText(jsonPath), StringComparison.Ordinal);

        using JsonDocument evidenceMap = JsonDocument.Parse(File.ReadAllText(evidenceMapPath));
        JsonElement mapRoot = evidenceMap.RootElement;
        Assert.Equal("deferred-readonly-candidate-evidence-map.v1", mapRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal(9, mapRoot.GetProperty("candidates").GetArrayLength());
        Assert.False(mapRoot.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(mapRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(mapRoot.GetProperty("canPromoteReleaseProof").GetBoolean());
        Assert.False(mapRoot.GetProperty("canDeleteDeferredRecords").GetBoolean());
    }

    private static string HashFile(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
    }

    private static string RunPowerShell(string scriptPath)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = PowerShellHost.ResolveExecutable(),
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}
