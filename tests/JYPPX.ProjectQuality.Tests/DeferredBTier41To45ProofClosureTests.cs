using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTier41To45ProofClosureTests
{
    [Fact]
    public void Btier41To46KeepSafeAlternativesDeferredHistoryAndPointerFreeWrappers()
    {
        string workPackage = ReadSource("artifacts", "interface-coverage", "deferred-btier-implementation-work-package.json");
        using JsonDocument document = JsonDocument.Parse(workPackage);
        Dictionary<string, JsonElement> workItems = document.RootElement.GetProperty("workItems")
            .EnumerateArray()
            .Where(static item => item.GetProperty("workItemId").GetString() is "btier-041" or "btier-042" or "btier-043" or "btier-044" or "btier-045" or "btier-046")
            .ToDictionary(static item => item.GetProperty("workItemId").GetString()!, static item => item);

        Assert.Equal(6, workItems.Count);
        AssertWorkItem(workItems["btier-041"], "IBinaryProtoBlob::getData", "trt8-legacy-caffe-binary-proto-copy", "trt8-binary-proto-blob-get-data-deferred");
        AssertWorkItem(workItems["btier-042"], "IBinaryProtoBlob::getDataType", "trt8-legacy-caffe-binary-proto-copy", "trt8-binary-proto-blob-get-data-type-deferred");
        AssertWorkItem(workItems["btier-043"], "IBinaryProtoBlob::getDimensions", "trt8-legacy-caffe-binary-proto-copy", "trt8-binary-proto-blob-get-dimensions-deferred");
        AssertWorkItem(workItems["btier-044"], "IUffParser::getUffRequiredVersionMajor", "trt8-legacy-uff-get-required-version", "trt8-uff-parser-get-uff-required-version-major-deferred");
        AssertWorkItem(workItems["btier-045"], "IUffParser::getUffRequiredVersionMinor", "trt8-legacy-uff-get-required-version", "trt8-uff-parser-get-uff-required-version-minor-deferred");
        AssertWorkItem(workItems["btier-046"], "IUffParser::getUffRequiredVersionPatch", "trt8-legacy-uff-get-required-version", "trt8-uff-parser-get-uff-required-version-patch-deferred");

        string legacyParserApi = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtLegacyParserDiagnostics.cs");
        string uffSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtLegacyUffRequiredVersionSnapshot.cs");
        string caffeSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtCaffeBinaryProtoSnapshot.cs");
        string managedSurface = string.Join('\n', legacyParserApi, uffSnapshot, caffeSnapshot);

        Assert.Contains("public static TensorRtLegacyUffRequiredVersionSnapshot GetUffRequiredVersion", legacyParserApi);
        Assert.Contains("public static TensorRtCaffeBinaryProtoSnapshot ReadCaffeBinaryProto", legacyParserApi);

        Assert.DoesNotContain("public IntPtr", managedSurface);
        Assert.DoesNotContain("public nint", managedSurface);
    }

    private static void AssertWorkItem(JsonElement item, string interfaceName, string safeAlternativeId, string deferredHistoryId)
    {
        Assert.Equal(interfaceName, item.GetProperty("interface").GetString());
        Assert.Equal("phase-1-safe-alternative-proof", item.GetProperty("phase").GetString());
        Assert.Equal("B - safe-alternative-or-alias", item.GetProperty("safetyTier").GetString());
        Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
        Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
        Assert.Contains(safeAlternativeId, item.GetProperty("safeAlternativeManifestIds").EnumerateArray().Select(static value => value.GetString()!));
        Assert.Contains(deferredHistoryId, item.GetProperty("deferredHistoryManifestIds").EnumerateArray().Select(static value => value.GetString()!));
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
