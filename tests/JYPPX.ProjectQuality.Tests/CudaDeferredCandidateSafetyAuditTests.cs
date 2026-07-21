using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaDeferredCandidateSafetyAuditTests
{
    [Fact]
    public void AuditExactlyCoversCurrentUniqueDeferredFunctionsAndKeepsUnsafeBoundary()
    {
        string coveragePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "cuda-runtime-interface-coverage.json");
        string auditPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "cuda-deferred-candidate-safety-audit.json");
        using JsonDocument coverage = JsonDocument.Parse(File.ReadAllText(coveragePath));
        using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));

        string[] deferredFunctions = coverage.RootElement.EnumerateArray()
            .Where(static row => row.GetProperty("ImplementationStatus").GetString() == "deferred-only")
            .Select(static row => row.GetProperty("Function").GetString()!)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();
        JsonElement root = audit.RootElement;
        string[] auditedFunctions = root.GetProperty("categories").EnumerateArray()
            .SelectMany(static category => category.GetProperty("functions").EnumerateArray())
            .Select(static item => item.GetString()!)
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(311, root.GetProperty("deferredRowCount").GetInt32());
        Assert.Equal(64, root.GetProperty("uniqueDeferredFunctionCount").GetInt32());
        Assert.Equal(0, root.GetProperty("immediateSafeCandidateCount").GetInt32());
        Assert.Equal(deferredFunctions, auditedFunctions);
        Assert.Equal(64, auditedFunctions.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(Sha256(coveragePath), root.GetProperty("sourceCoverageSha256").GetString(), ignoreCase: true);
        Assert.All(root.GetProperty("categories").EnumerateArray(), static category => Assert.Equal("keep-deferred", category.GetProperty("decision").GetString()));
        Assert.False(root.GetProperty("publicSurfaceBoundary").GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("publicSurfaceBoundary").GetProperty("canPublishPublicly").GetBoolean());
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }
}
