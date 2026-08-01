using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class BridgeCallbackRuntimeEvidenceSummaryTests
{
    private static readonly string[] RuntimeKeys =
    [
        "win-x64-trt10.11-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda12.9-cudnn9.22",
    ];

    [Fact]
    public void ExporterAndReleaseBundleKeepLocalCallbackEvidenceInItsOwnScope()
    {
        string exporter = ReadSource("eng", "Export-BridgeCallbackRuntimeEvidenceSummary.ps1");
        string releaseBundle = ReadSource("eng", "Export-ReleaseEvidenceBundle.ps1");

        foreach (string marker in new[]
        {
            "bridge-callback-runtime-evidence-summary",
            "allRequiredLocalPackageCallbackProofObserved",
            "sourceReportsContainRuntimeExecutionProof",
            "sourceReportsContainLocalPackageCallbackRuntimeProof",
            "callback-state-not-observed",
            "callback-state-incoherent",
            "callback-state-not-pointer-free",
            "callback-not-invoked",
            "callback-failure-count-nonzero",
            "callback-inflight-count-nonzero",
            "public-package-scope-must-remain-false",
            "post-publish-scope-must-remain-false",
            "isRuntimeExecutionProof = $false",
            "isPackageConsumerRuntimeProof = $false",
            "isPublicPackageProof = $false",
            "isPostPublishProof = $false",
            "canPromoteRuntimeProof = $false",
            "canPublishPublicly = $false",
            "canCloseReleaseIssue = $false",
        })
        {
            Assert.Contains(marker, exporter, StringComparison.Ordinal);
        }

        Assert.Contains("Export-BridgeCallbackRuntimeEvidenceSummary.ps1", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("bridgeCallbackRuntimeEvidenceSummary = $bridgeCallbackRuntimeEvidenceSummary", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("bridgeCallbackRuntimeEvidenceSummarySourceLocalCallbackProof", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("summary-runtime-proof=``$bridgeCallbackRuntimeEvidenceSummaryIsRuntimeProof``", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("public-package-proof=``$bridgeCallbackRuntimeEvidenceSummaryIsPublicPackageProof``", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("post-publish-proof=``$bridgeCallbackRuntimeEvidenceSummaryIsPostPublishProof``", releaseBundle, StringComparison.Ordinal);
    }

    [Fact]
    public void ExporterObservesTwoValidLocalPackageCallbackReportsWithoutPromotingTheSummary()
    {
        using TemporaryEvidenceRoot root = new();
        foreach (string runtimeKey in RuntimeKeys)
        {
            root.WriteReport(runtimeKey, publicPackageProof: false);
        }

        ProcessResult process = root.RunExporter();
        Assert.True(process.ExitCode == 0, process.Output);

        using JsonDocument document = root.ReadSummary();
        JsonElement summary = document.RootElement;
        Assert.Equal("local-package-callback-runtime-observed", summary.GetProperty("summaryState").GetString());
        Assert.Equal("local-package", summary.GetProperty("evidenceScope").GetString());
        Assert.Equal(2, summary.GetProperty("requiredRuntimeKeyCount").GetInt32());
        Assert.Equal(2, summary.GetProperty("observedRuntimeKeyCount").GetInt32());
        Assert.True(summary.GetProperty("allRequiredLocalPackageCallbackProofObserved").GetBoolean());
        Assert.True(summary.GetProperty("sourceReportsContainRuntimeExecutionProof").GetBoolean());
        Assert.True(summary.GetProperty("sourceReportsContainLocalPackageCallbackRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(summary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("isPublicPackageProof").GetBoolean());
        Assert.False(summary.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(summary.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(summary.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(0, summary.GetProperty("findingCount").GetInt32());

        JsonElement[] rows = summary.GetProperty("rows").EnumerateArray().ToArray();
        Assert.Equal(2, rows.Length);
        Assert.All(rows, static row =>
        {
            Assert.Equal("local-package-callback-runtime-observed", row.GetProperty("status").GetString());
            Assert.True(row.GetProperty("valid").GetBoolean());
            Assert.True(row.GetProperty("callbackStateObserved").GetBoolean());
            Assert.True(row.GetProperty("callbackStateCoherent").GetBoolean());
            Assert.True(row.GetProperty("callbackStatePointerFree").GetBoolean());
            Assert.Equal(1, row.GetProperty("invocationCount").GetInt64());
            Assert.Equal(0, row.GetProperty("failureCount").GetInt64());
            Assert.Equal(0, row.GetProperty("inFlightCallbackCount").GetInt64());
            Assert.True(row.GetProperty("localPackageProof").GetBoolean());
            Assert.False(row.GetProperty("publicPackageProof").GetBoolean());
            Assert.False(row.GetProperty("postPublishProof").GetBoolean());
        });
    }

    [Fact]
    public void ExporterRejectsMissingAndPublicScopeReportsWithoutPromotingAnyProof()
    {
        using TemporaryEvidenceRoot root = new();
        root.WriteReport(RuntimeKeys[0], publicPackageProof: true);

        ProcessResult process = root.RunExporter();
        Assert.True(process.ExitCode == 0, process.Output);

        using JsonDocument document = root.ReadSummary();
        JsonElement summary = document.RootElement;
        Assert.Equal("blocked-local-package-callback-runtime-evidence-incomplete", summary.GetProperty("summaryState").GetString());
        Assert.Equal(0, summary.GetProperty("observedRuntimeKeyCount").GetInt32());
        Assert.False(summary.GetProperty("allRequiredLocalPackageCallbackProofObserved").GetBoolean());
        Assert.False(summary.GetProperty("sourceReportsContainRuntimeExecutionProof").GetBoolean());
        Assert.False(summary.GetProperty("sourceReportsContainLocalPackageCallbackRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(summary.GetProperty("isPublicPackageProof").GetBoolean());
        Assert.False(summary.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(summary.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(summary.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(2, summary.GetProperty("findingCount").GetInt32());

        JsonElement[] rows = summary.GetProperty("rows").EnumerateArray().ToArray();
        JsonElement invalid = rows.Single(row => row.GetProperty("runtimeKey").GetString() == RuntimeKeys[0]);
        JsonElement missing = rows.Single(row => row.GetProperty("runtimeKey").GetString() == RuntimeKeys[1]);
        Assert.Equal("invalid-local-package-callback-runtime-evidence", invalid.GetProperty("status").GetString());
        Assert.Contains(
            invalid.GetProperty("findings").EnumerateArray(),
            static finding => finding.GetString() == "public-package-scope-must-remain-false");
        Assert.Equal("missing-report", missing.GetProperty("status").GetString());
        Assert.Contains(
            missing.GetProperty("findings").EnumerateArray(),
            static finding => finding.GetString() == "missing-report");
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }

    private sealed class TemporaryEvidenceRoot : IDisposable
    {
        private readonly string _root = Path.Combine(Path.GetTempPath(), "jyppx-bridge-callback-summary-" + Guid.NewGuid().ToString("N"));

        internal TemporaryEvidenceRoot()
        {
            ReportRoot = Path.Combine(_root, "reports");
            OutputRoot = Path.Combine(_root, "output");
            Directory.CreateDirectory(ReportRoot);
            Directory.CreateDirectory(OutputRoot);
        }

        internal string ReportRoot { get; }

        internal string OutputRoot { get; }

        internal void WriteReport(string runtimeKey, bool publicPackageProof)
        {
            string directory = Path.Combine(ReportRoot, runtimeKey);
            Directory.CreateDirectory(directory);
            var payload = new
            {
                generatedAtUtc = DateTime.UtcNow.ToString("O"),
                sourceRuntimeKey = runtimeKey,
                proofClassification = "compatible-host-bridge-package-runtime",
                smokeStatus = "passed",
                isRuntimeExecutionProof = true,
                isPackageConsumerRuntimeProof = false,
                isLocalPackageDebugListenerCallbackRuntimeProof = true,
                canPromoteRuntimeProof = false,
                canPublishPublicly = false,
                canCloseReleaseIssue = false,
                callbackStateSnapshot = new
                {
                    observed = true,
                    complete = false,
                    coherent = true,
                    pointerFree = true,
                    lastStatus = "RuntimeError",
                    lastOperation = "snapshot-debug-listener-interface-info-partial",
                },
                debugListenerCallback = new
                {
                    status = "passed",
                    attached = true,
                    nativeVTableInstalled = true,
                    invocationCount = 1,
                    failureCount = 0,
                    inFlightCallbackCount = 0,
                    detached = true,
                    clearReturned = true,
                    isRealCallbackRuntimeProof = true,
                    isLocalPackageCallbackRuntimeProof = true,
                },
                proofScopes = new
                {
                    sourceTree = new { isProof = false },
                    localPackage = new { isProof = true },
                    publicPackage = new { isProof = publicPackageProof },
                    postPublish = new { isProof = false },
                },
            };
            string path = Path.Combine(directory, "bridge-package-runtime-consumer-proof.json");
            File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
        }

        internal ProcessResult RunExporter()
        {
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-BridgeCallbackRuntimeEvidenceSummary.ps1");
            ProcessStartInfo startInfo = new()
            {
                FileName = OperatingSystem.IsWindows() ? "powershell" : "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            startInfo.ArgumentList.Add("-NoProfile");
            if (OperatingSystem.IsWindows())
            {
                startInfo.ArgumentList.Add("-ExecutionPolicy");
                startInfo.ArgumentList.Add("Bypass");
            }
            startInfo.ArgumentList.Add("-File");
            startInfo.ArgumentList.Add(script);
            startInfo.ArgumentList.Add("-ReportRoot");
            startInfo.ArgumentList.Add(ReportRoot);
            startInfo.ArgumentList.Add("-OutputRoot");
            startInfo.ArgumentList.Add(OutputRoot);
            startInfo.ArgumentList.Add("-RepositoryRoot");
            startInfo.ArgumentList.Add(RepositoryPaths.Root);

            using Process process = Process.Start(startInfo)!;
            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();
            return new ProcessResult(process.ExitCode, stdout + stderr);
        }

        internal JsonDocument ReadSummary()
        {
            string path = Path.Combine(OutputRoot, "bridge-callback-runtime-evidence-summary.json");
            Assert.True(File.Exists(path), "Summary was not written.");
            return JsonDocument.Parse(File.ReadAllText(path));
        }

        public void Dispose()
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
    }

    private sealed record ProcessResult(int ExitCode, string Output);
}
