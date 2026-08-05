using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class StreamReaderLocalPackageConsumerTests
{
    [Fact]
    public void PublicSurfaceIsOwnerSafeAndPreservesExistingDeserializeOverloads()
    {
        string reader = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Callbacks", "Serialization", "TensorRtStreamReader.cs");
        string snapshot = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Callbacks", "Serialization", "TensorRtStreamReaderRuntimeSnapshot.cs");
        string runtime = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.cs");
        string engine = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");

        Assert.Contains("namespace JYPPX.TensorRtSharp;", reader, StringComparison.Ordinal);
        Assert.Contains("public TensorRtStreamReader(TensorRtApiLine line, byte[] serializedEngine)", reader, StringComparison.Ordinal);
        Assert.Contains("public TensorRtStreamReader(TensorRtApiLine line, Stream serializedEngineStream)", reader, StringComparison.Ordinal);
        Assert.Contains("_nativeHandle = NativeBridgeApi.CreateStreamReaderV2Owner(line, serializedEngine);", reader, StringComparison.Ordinal);
        Assert.Contains("_nativeHandle = new SafeTensorRtObjectHandle();", reader, StringComparison.Ordinal);
        Assert.Contains("checked", reader, StringComparison.Ordinal);
        Assert.Contains("BeginDeserializeBorrower", reader, StringComparison.Ordinal);
        Assert.Contains("RetainEngineBorrower", reader, StringComparison.Ordinal);
        Assert.Contains("ReleaseEngineBorrower", reader, StringComparison.Ordinal);
        Assert.Contains("public TensorRtEngine Deserialize(TensorRtStreamReader streamReader)", runtime, StringComparison.Ordinal);
        Assert.Contains("public TensorRtEngine Deserialize(byte[] serializedEngine)", runtime, StringComparison.Ordinal);
        Assert.Contains("public TensorRtEngine Deserialize(Stream serializedEngineStream)", runtime, StringComparison.Ordinal);
        Assert.Contains("_streamReaderKeepAlive?.RetainEngineBorrower(line);", engine, StringComparison.Ordinal);
        Assert.Contains("_streamReaderKeepAlive?.ReleaseEngineBorrower();", engine, StringComparison.Ordinal);

        string publicSurface = reader + snapshot;
        Assert.DoesNotMatch(@"public\s+[^\r\n]*(IntPtr|UIntPtr|SafeHandle|nint|nuint)", publicSurface);
    }

    [Fact]
    public void NativeOwnerIsVersionGuardedNoncopyableAndNoThrow()
    {
        string types = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string implementation = ReadSource(
            "native", "src", "tensorrt", "common", "stream_reader_callback_owner.inc");
        string objectSource = ReadSource("native", "src", "tensorrt", "common", "object.cpp");
        string trt10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_STREAM_READER_CALLBACK_OWNER = 29", types, StringComparison.Ordinal);
        Assert.Contains("typedef struct JYPPX_TensorRtStreamReaderOwnerInfo", types, StringComparison.Ordinal);
        Assert.Contains("class StreamReaderCallbackOwner final : public nvinfer1::IStreamReaderV2", implementation, StringComparison.Ordinal);
        Assert.Contains("read(void* destination, const int64_t nb_bytes, cudaStream_t stream) noexcept override", implementation, StringComparison.Ordinal);
        Assert.Contains("seek(const int64_t offset, const nvinfer1::SeekPosition where) noexcept override", implementation, StringComparison.Ordinal);
        Assert.Contains("std::is_nothrow_destructible<StreamReaderCallbackOwner>", implementation, StringComparison.Ordinal);
        Assert.Contains("std::is_copy_constructible<StreamReaderCallbackOwner>", implementation, StringComparison.Ordinal);
        Assert.Contains("cudaPointerGetAttributes", implementation, StringComparison.Ordinal);
        Assert.Contains("deserialize_stream_reader_with_seh_guard", implementation, StringComparison.Ordinal);
        Assert.Contains("stream-reader-callback-owner", objectSource, StringComparison.Ordinal);
        Assert.Contains("#define JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR 10", trt10, StringComparison.Ordinal);
        Assert.Contains("#define JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR 11", trt11, StringComparison.Ordinal);
    }

    [Fact]
    public void ManifestsAndGeneratedBindingsExposeExactlyThreeEntriesPerSupportedLine()
    {
        foreach (string line in new[] { "10", "11" })
        {
            string manifestPath = Path.Combine(
                RepositoryPaths.Root,
                "native", "manifests", "tensorrt", "v" + line,
                "trt" + line + "-stream-reader-v2-callback-owner.manifest.json");
            using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
            JsonElement[] apis = manifest.RootElement.GetProperty("apis").EnumerateArray().ToArray();
            Assert.Equal(3, apis.Length);
            Assert.Contains(apis, api => api.GetProperty("entryPoint").GetString() == $"jyppx_trt{line}_stream_reader_v2_owner_create");
            Assert.Contains(apis, api => api.GetProperty("entryPoint").GetString() == $"jyppx_trt{line}_stream_reader_v2_owner_get_info");
            Assert.Contains(apis, api => api.GetProperty("entryPoint").GetString() == $"jyppx_trt{line}_runtime_deserialize_stream_reader_v2");
        }

        string generated = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated",
            "NativeMethodsTensorRt.Generated.g.cs");
        Assert.Equal(2, Regex.Matches(generated, "stream_reader_v2_owner_create").Count);
        Assert.Equal(2, Regex.Matches(generated, "stream_reader_v2_owner_get_info").Count);
        Assert.Equal(2, Regex.Matches(generated, "runtime_deserialize_stream_reader_v2").Count);
    }

    [Fact]
    public void ExternalConsumerAndHarnessEnforceRealRuntimeAndPackageIsolation()
    {
        string template = ReadSource(
            "samples", "StreamReader.PackageConsumer", "StreamReader.PackageConsumer.csproj.template");
        string program = ReadSource("samples", "StreamReader.PackageConsumer", "Program.cs");
        string wrapper = ReadSource("eng", "Test-StreamReaderLocalPackageConsumer.ps1");
        string harness = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");

        Assert.Equal(2, Regex.Matches(template, "<PackageReference ").Count);
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("HintPath", template, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "using JYPPX.TensorRtSharp;",
            "using JYPPX.TensorRtSharp.Shared.Interop;",
            "TensorRtEnvironmentProbe.GetCurrent()",
            "TensorRtStreamReader reader = new(line, mutableSource)",
            "Array.Clear(mutableSource, 0, mutableSource.Length)",
            "runtime.Deserialize(reader)",
            "RunDeferredDisposeCase",
            "RunTruncatedInputCase",
            "PublicSnapshotExposesNativePointer",
            "StreamReaderRealRuntime=Passed",
            "StreamReaderPackageConsumer Passed=True"
        })
        {
            Assert.Contains(required, program, StringComparison.Ordinal);
        }

        Assert.Contains("-Scenario StreamReader @PSBoundParameters", wrapper, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "StreamReader.PackageConsumer",
            "--stream-reader-runtime-smoke-only",
            "StreamReaderRealRuntime=Passed",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "consumerBridgeMatchesPackage",
            "$attemptCount -ne 2",
            "$truncatedFailedCount -ne 1",
            "$pointerExposed",
            "$streamWriterVerified",
            "PublicPackageProof=False PerformsPublish=False"
        })
        {
            Assert.Contains(required, harness, StringComparison.Ordinal);
        }

        string combined = wrapper + harness;
        Assert.DoesNotContain("dotnet nuget push", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release create", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("git tag", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EvidenceMatchesArtifactsRuntimeNegativesAndBoundary()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples", "assets", "stream-reader-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument document = JsonDocument.Parse(evidenceText);
        JsonElement root = document.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-stream-reader-v2-runtime", root.GetProperty("evidenceKind").GetString());
        Assert.Equal("passed-local-package-consumer-runtime", root.GetProperty("validationState").GetString());
        Assert.Equal("10.11.0", root.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("12.9", root.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal(2, root.GetProperty("packages").GetArrayLength());

        JsonElement model = root.GetProperty("model");
        Assert.Equal("not-applicable", model.GetProperty("acquisition").GetString());
        Assert.Equal("not-applicable", model.GetProperty("conversion").GetString());
        Assert.False(model.GetProperty("externalModelRequired").GetBoolean());
        Assert.False(model.GetProperty("outerModelsDirectoryUsed").GetBoolean());

        JsonElement isolation = root.GetProperty("isolation");
        Assert.True(isolation.GetProperty("workspaceOutsideRepository").GetBoolean());
        Assert.True(isolation.GetProperty("packageReferenceOnly").GetBoolean());
        Assert.False(isolation.GetProperty("projectReference").GetBoolean());
        Assert.False(isolation.GetProperty("sourceTreeBinary").GetBoolean());
        Assert.True(isolation.GetProperty("consumerBridgeMatchesPackage").GetBoolean());
        Assert.Equal(0, isolation.GetProperty("vendorRuntimeBinaryCountInPackages").GetInt32());
        Assert.Equal(0, isolation.GetProperty("vendorRuntimeBinaryCountInConsumerOutput").GetInt32());

        JsonElement positive = root.GetProperty("positive");
        Assert.True(positive.GetProperty("passed").GetBoolean());
        Assert.Equal(2UL, positive.GetProperty("deserializeAttemptCount").GetUInt64());
        Assert.Equal(2UL, positive.GetProperty("successfulDeserializeCount").GetUInt64());
        Assert.Equal(0UL, positive.GetProperty("failedDeserializeCount").GetUInt64());
        Assert.True(positive.GetProperty("readCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("seekCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("bytesRead").GetUInt64() > 0UL);
        Assert.Equal(0UL, positive.GetProperty("failureCount").GetUInt64());
        Assert.True(positive.GetProperty("metadataMatched").GetBoolean());
        Assert.True(positive.GetProperty("sourceCopied").GetBoolean());
        Assert.True(positive.GetProperty("reusable").GetBoolean());
        Assert.False(positive.GetProperty("nativePointerExposed").GetBoolean());
        Assert.True(positive.GetProperty("realCallbackRuntime").GetBoolean());

        JsonElement lifecycle = root.GetProperty("deferredDispose");
        Assert.True(lifecycle.GetProperty("passed").GetBoolean());
        Assert.True(lifecycle.GetProperty("disposeDeferred").GetBoolean());
        Assert.True(lifecycle.GetProperty("rejectsNewDeserialize").GetBoolean());
        Assert.True(lifecycle.GetProperty("releasedAfterEngine").GetBoolean());

        JsonElement truncated = root.GetProperty("truncatedInputNegative");
        Assert.True(truncated.GetProperty("passed").GetBoolean());
        Assert.True(truncated.GetProperty("deserializeFailed").GetBoolean());
        Assert.Equal(1UL, truncated.GetProperty("attemptCount").GetUInt64());
        Assert.Equal(1UL, truncated.GetProperty("failedCount").GetUInt64());
        Assert.True(truncated.GetProperty("failureCount").GetUInt64() > 0UL);

        JsonElement versions = root.GetProperty("versionBoundary");
        Assert.True(versions.GetProperty("tensorRt10Verified").GetBoolean());
        Assert.False(versions.GetProperty("tensorRt11RuntimeVerified").GetBoolean());
        Assert.True(versions.GetProperty("tensorRt8Rejected").GetBoolean());
        Assert.True(versions.GetProperty("legacyStreamReaderDeferred").GetBoolean());
        Assert.False(versions.GetProperty("streamWriterVerified").GetBoolean());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing stream reader artifact: {relativePath}");
            Assert.Equal(artifact.Value.GetProperty("length").GetInt64(), new FileInfo(path).Length);
            string hash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            Assert.Equal(artifact.Value.GetProperty("sha256").GetString(), hash);
        }

        JsonElement boundary = root.GetProperty("boundary");
        Assert.True(boundary.GetProperty("isLocalPackageConsumerRuntimeEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isPublicPackageConsumerProof").GetBoolean());
        Assert.False(boundary.GetProperty("isReleaseProof").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.DoesNotMatch(@"[A-Za-z]:\\", evidenceText);
    }

    [Fact]
    public void TutorialIsCompletePathFreeAndUsesTheRealRuntimeScreenshot()
    {
        string article = ReadSource(
            "docs", "articles", "zh-cn", "stream-reader-local-package-consumer-tutorial.md");
        string toc = ReadSource("docs", "toc.yml");

        foreach (string required in new[]
        {
            "## 1. 项目与功能背景",
            "## 2. 依赖与包职责",
            "## 3. 模型获取与转换说明",
            "官方获取方式：不适用",
            "ONNX 转换方式：不适用",
            "图像识别结果：不适用",
            "## 4. native owner 与 ABI 设计",
            "## 5. C# 所有权与公开接口",
            "## 7. 仓库外消费者如何隔离",
            "## 8. 正例：真实 read、seek 与顺序复用",
            "## 9. 生命周期负例：提前 Dispose",
            "## 10. 数据负例：截断 plan 必须失败",
            "## 12. 本机执行结果",
            "stream-reader-local-package-consumer-terminal.png",
            "Attempts=2 Success=2 Reads=10 Seeks=10",
            "TruncatedInput=Passed DeserializeFailed=True Attempts=1 Failures=2",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("stream-reader-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
