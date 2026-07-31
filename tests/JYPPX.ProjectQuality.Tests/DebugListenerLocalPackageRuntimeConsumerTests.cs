using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerLocalPackageRuntimeConsumerTests
{
    [Fact]
    public void RuntimeConsumerExecutesRealDebugListenerThroughPackageReferencesOnly()
    {
        string script = ReadScript();

        Assert.Contains("network.MarkDebugTensor(outputTensor)", script, StringComparison.Ordinal);
        Assert.Contains("new TensorRtDebugListenerCallbackOwner(", script, StringComparison.Ordinal);
        Assert.Contains("context.SetDebugListener(debugListenerOwner)", script, StringComparison.Ordinal);
        Assert.Contains("context.SetTensorDebugState(\"output\", true)", script, StringComparison.Ordinal);
        Assert.Contains("bindings.EnqueueAsync(stream, synchronize: true", script, StringComparison.Ordinal);
        Assert.Contains("context.ClearDebugListener()", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackEvidenceScope=local-package", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackRuntime=", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackBorrowedPointerExposed=", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackDetachCount=", script, StringComparison.Ordinal);
        Assert.Contains("<PackageReference Include=\"$($managedPackage.Id)\"", script, StringComparison.Ordinal);
        Assert.Contains("<PackageReference Include=\"$($bridgePackage.Id)\"", script, StringComparison.Ordinal);
        Assert.DoesNotContain("<ProjectReference Include=", script, StringComparison.Ordinal);
        Assert.DoesNotContain("<Reference Include=", script, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeConsumerPromotionIsFailClosedForMissingOrInvalidCallbackMarkers()
    {
        string script = ReadScript();

        Assert.Contains("$debugListenerInvocationCountParsed -and $debugListenerInvocationCount -gt 0", script, StringComparison.Ordinal);
        Assert.Contains("$debugListenerFailureCountParsed -and $debugListenerFailureCount -eq 0", script, StringComparison.Ordinal);
        Assert.Contains("$debugListenerInFlightCallbackCountParsed -and $debugListenerInFlightCallbackCount -eq 0", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackMetadataCopied=\") -eq \"True\"", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackBorrowedPointerExposed=\") -eq \"False\"", script, StringComparison.Ordinal);
        Assert.Contains("$debugListenerDetachCountParsed -and $debugListenerDetachCount -gt 0", script, StringComparison.Ordinal);
        Assert.Contains("DebugListenerCallbackIsRealRuntimeProof=\") -eq \"True\"", script, StringComparison.Ordinal);
        Assert.Contains("-not $debugListenerCallbackRequired -or $debugListenerCallbackRuntimePassed", script, StringComparison.Ordinal);
        Assert.Contains("isLocalPackageCallbackRuntimeProof = $isLocalPackageDebugListenerCallbackRuntimeProof", script, StringComparison.Ordinal);
        Assert.Contains("-not $SkipInstalledVendorAssetHashing.IsPresent", script, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeConsumerReportKeepsEvidenceScopesSeparate()
    {
        string script = ReadScript();

        Assert.Contains("proofScopes = [ordered]@{", script, StringComparison.Ordinal);
        Assert.Contains("evidenceScope = \"source-tree\"", script, StringComparison.Ordinal);
        Assert.Contains("evidenceScope = \"local-package\"", script, StringComparison.Ordinal);
        Assert.Contains("evidenceScope = \"public-package\"", script, StringComparison.Ordinal);
        Assert.Contains("evidenceScope = \"post-publish\"", script, StringComparison.Ordinal);
        Assert.Contains("usesDirectAssemblyReference = $usesDirectAssemblyReference", script, StringComparison.Ordinal);
        Assert.Contains("usesRepositorySourceProbe = $usesRepositorySourceProbe", script, StringComparison.Ordinal);
        Assert.Contains("isLocalPackageDebugListenerCallbackRuntimeProof = $isLocalPackageDebugListenerCallbackRuntimeProof", script, StringComparison.Ordinal);
        Assert.Contains("isPublicCleanPackageConsumerProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("isPostPublishCleanConsumerProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
    }

    private static string ReadScript()
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-BridgePackageRuntimeConsumer.ps1"));
    }
}
