using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PackageConsumerSmokeEvidenceTests
{
    [Fact]
    public void SmokeReadmeDocumentsParserRegistryAndRuntimeMarkersAsNonSubstituteEvidence()
    {
        string readme = ReadSource("smoke", "README.md");

        Assert.Contains("OnnxToEngineSmokeRunner", readme);
        Assert.Contains("BuilderPluginRegistry", readme);
        Assert.Contains("GlobalPluginRegistryLookup", readme);
        Assert.Contains("ParserDiagnosticSnapshot=", readme);
        Assert.Contains("ParserDiagnosticSummary=", readme);
        Assert.Contains("ParserRefitterDiagnosticSnapshot=", readme);
        Assert.Contains("ParserRefitterDiagnosticSummary=", readme);
        Assert.Contains("RuntimeDiagnosticSnapshot=", readme);
        Assert.Contains("RefitterDiagnosticSnapshot=", readme);
        Assert.Contains("ParserModelSupport", readme);
        Assert.Contains("ParserUsedVCPluginLibraries Count=", readme);
        Assert.Contains("LayerOutputIdentity=", readme);
        Assert.Contains("StreamRoundTrip=True", readme);
        Assert.Contains("InferenceBindings Execution=", readme);
        Assert.Contains("not a substitute for a clean package-consumer runtime proof, Linux runner proof, or real external-model proof", readme);
        Assert.Contains("Skipped=True Reason=", readme);
        Assert.Contains("must not be promoted to release runtime proof", readme);
    }

    [Fact]
    public void PackageConsumerScriptCanRecordSmokeFailureOnlyWhenExplicitlyAllowed()
    {
        string script = ReadSource("eng", "Test-PackageConsumer.ps1");

        Assert.Contains("[switch]$RunSmoke", script);
        Assert.Contains("[switch]$AllowSmokeFailure", script);
        Assert.Contains("SmokeRequested", script);
        Assert.Contains("SmokeExitCode", script);
        Assert.Contains("SmokeCommand", script);
        Assert.Contains("SmokeDiagnostic", script);
        Assert.Contains("SmokeOutputLines", script);
        Assert.Contains("SmokeFailureAllowed", script);
        Assert.Contains("ConsumerBuildConfiguration = \"Release\"", script);
        Assert.Contains("RestoreSucceeded = $true", script);
        Assert.Contains("BuildSucceeded = $true", script);
        Assert.Contains("PackageConsumerValidationSucceeded = $true", script);
        Assert.Contains("EvidenceKind = $packageConsumerEvidenceKind", script);
        Assert.Contains("IsRuntimeExecutionEvidence = $isRuntimeExecutionEvidence", script);
        Assert.Contains("CanPromoteRuntimeProof = $false", script);
        Assert.Contains("CanPublishPublicly = $false", script);
        Assert.Contains("CanCloseReleaseIssue = $false", script);
        Assert.Contains("RuntimeSmokeClassification = $runtimeSmokeClassification", script);
        Assert.Contains("IsDependencyProbeOnly = $isDependencyProbeOnly", script);
        Assert.Contains("IsRealCallbackRuntimeProof = $isRealCallbackRuntimeProof", script);
        Assert.Contains("package-consumer-native-copy", script);
        Assert.Contains("full-runtime-package-consumer-smoke-driver-blocked", script);
        Assert.Contains("runtime-smoke-driver-blocked", script);
        Assert.Contains("Evidence Classification", script);
        Assert.Contains("IsDependencyProbeOnly=True", script);
        Assert.Contains("not as runtime execution proof", script);
        Assert.Contains("RealCallbackRuntimeEvidence", script);
        Assert.Contains("New-RealCallbackRuntimeEvidenceFromSmoke", script);
        Assert.Contains("[AllowEmptyString()]", script);
        Assert.Contains("[string]$Text = \"\"", script);
        Assert.Contains("[string]::IsNullOrEmpty($Text)", script);
        Assert.Contains("RealCallbackRuntimeRequiredSmokeMarkers", script);
        Assert.Contains("Status = \"not-present\"", script);
        Assert.Contains("\"blocked-by-cuda-driver\" { \"blocked-by-cuda-driver\"; break }", script);
        Assert.Contains("\"blocked-by-application-control\" { \"blocked-by-application-control\"; break }", script);
        Assert.Contains("\"failed\" { \"blocked\"; break }", script);
        Assert.Contains("Status = $status", script);
        Assert.Contains("EvidenceKind = if ($isReady) { \"real-callback-runtime\" } else { \"incomplete-real-callback-runtime\" }", script);
        Assert.Contains("IsRealCallbackRuntimeProof = $isReady", script);
        Assert.Contains("MissingSmokeMarkers", script);
        Assert.Contains("MatchedSmokeLines", script);
        Assert.Contains("failed", script);
        Assert.Contains("Test-CudaDriverRuntimeCompatibilityBlock", script);
        Assert.Contains("blocked-by-cuda-driver", script);
        Assert.Contains("CUDA error 35", script);
        Assert.Contains("if (-not $AllowSmokeFailure.IsPresent)", script);
        Assert.Contains("throw \"$smokeDiagnostic", script);
        Assert.Contains("Package consumer smoke failed but -AllowSmokeFailure was provided", script);
        Assert.Contains("Real callback runtime evidence is not inferred from `SmokeResult=passed`", script);
        Assert.Contains("output-allocator-attach-detach-design-gate", script);
        Assert.Contains("output-allocator-runtime-proof-precheck", script);
        Assert.Contains("debug-listener-attach-detach-design-gate", script);
        Assert.Contains("debug-listener-runtime-proof-precheck", script);
        Assert.Contains("precheck/dependency-probe evidence only", script);
        Assert.Contains("EvidenceKind=real-callback-runtime", script);
        Assert.Contains("RealCallbackRuntime=True", script);
        Assert.Contains("CallbackKind", script);
        Assert.Contains("TensorRtLine", script);
        Assert.Contains("CudaLine", script);
        Assert.Contains("RuntimePackageKey", script);
        Assert.Contains("OwnerId", script);
        Assert.Contains("InvocationCount", script);
        Assert.Contains("AllocationCount", script);
        Assert.Contains("ReleaseCount", script);
        Assert.Contains("FailureCount", script);
        Assert.Contains("InFlightCallbackCount", script);
        Assert.Contains("FullPackageConsumerReport", script);
        Assert.Contains("isRealCallbackRuntimeProof=true", script);
        Assert.Contains("Real callback runtime evidence: $($realCallbackRuntimeEvidence.Status) proof=$($realCallbackRuntimeEvidence.IsRealCallbackRuntimeProof)", script);
        Assert.Contains("Callback runtime evidence | Runtime proof", script);
        Assert.Contains("RealCallbackRuntimeEvidence.Status", script);
        Assert.Contains("only `ready` with `IsRealCallbackRuntimeProof=True` can be promoted by readiness", script);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
