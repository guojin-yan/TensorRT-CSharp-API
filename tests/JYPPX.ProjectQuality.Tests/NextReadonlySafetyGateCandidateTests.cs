using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class NextReadonlySafetyGateCandidateTests
{
    [Fact]
    public void CandidateListIncludesNextReadonlySafetyGateBatch()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        string text = File.ReadAllText(candidatePath);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement readonlyDiagnostics = document.RootElement.GetProperty("groups").GetProperty("readonlyDiagnostics");

        foreach (string candidateId in new[]
                 {
                     "stream-io-interface-info-design-004",
                     "calibrator-interface-info-design-004",
                     "logger-finder-metadata-design-004",
                     "debug-listener-interface-info-design-004",
                     "allocator-interface-info-design-004",
                 })
        {
            AssertCandidateExists(readonlyDiagnostics, candidateId);
        }

        foreach (string term in new[]
                 {
                     "IStreamReader::getInterfaceInfo",
                     "IStreamReaderV2::getInterfaceInfo",
                     "IStreamWriter::getInterfaceInfo",
                     "IInt8EntropyCalibrator::getInterfaceInfo",
                     "IInt8EntropyCalibrator2::getInterfaceInfo",
                     "IInt8LegacyCalibrator::getInterfaceInfo",
                     "IInt8MinMaxCalibrator::getInterfaceInfo",
                     "ILoggerFinder::getInterfaceInfo",
                     "IDebugListener::getInterfaceInfo",
                     "IDebugListener::processDebugTensor",
                     "IGpuAllocator::getInterfaceInfo",
                     "IGpuAsyncAllocator::getInterfaceInfo",
                     "IOutputAllocator::getInterfaceInfo",
                     "read/seek/write callbacks remain deferred",
                     "calibration callbacks remain deferred",
                     "allocation callbacks remain deferred",
                     "not runtime proof",
                 })
        {
            Assert.Contains(term, text, StringComparison.Ordinal);
        }
    }

    private static void AssertCandidateExists(JsonElement candidates, string candidateId)
    {
        foreach (JsonElement candidate in candidates.EnumerateArray())
        {
            if (candidate.GetProperty("candidateId").GetString() == candidateId)
            {
                Assert.True(candidate.GetProperty("implementationEvidence").TryGetProperty("ownershipBoundary", out _), candidateId);
                return;
            }
        }

        throw new InvalidOperationException("Candidate not found: " + candidateId);
    }
}
