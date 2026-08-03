using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using JYPPX.CudaSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaRuntimeCompilationOwnerTests
{
    [Fact]
    public void ManagedInputsRejectEmbeddedNulAndDuplicates()
    {
        Assert.Throws<ArgumentException>(() => new CudaRtcProgramSource("kernel\0source"));
        Assert.Throws<ArgumentException>(() => new CudaRtcProgramSource(
            "source",
            headers: new[] { new CudaRtcHeader("same.h", "a"), new CudaRtcHeader("same.h", "b") }));
        Assert.Throws<ArgumentException>(() => new CudaRtcProgramSource(
            "source",
            nameExpressions: new[] { "&kernel", "&kernel" }));
        Assert.Throws<ArgumentException>(() => new CudaRtcCompileOptions(additionalOptions: new[] { "--use_fast_math", "--use_fast_math" }));
        Assert.Throws<ArgumentException>(() => new CudaRtcCompileOptions(targetArchitecture: "invalid-target"));
    }

    [Fact]
    public void PublicRtcSurfaceIsPointerFreeAndUsesCopiedArtifacts()
    {
        Type[] rtcTypes =
        {
            typeof(CudaRtcCompiler),
            typeof(CudaRtcProgram),
            typeof(CudaRtcProgramSource),
            typeof(CudaRtcCompileOptions),
            typeof(CudaRtcCompilationResult),
            typeof(CudaRtcArtifact),
            typeof(CudaRtcCapability),
            typeof(CudaKernelLibrary),
            typeof(CudaKernelLaunch),
            typeof(CudaKernelArgument),
            typeof(CudaKernelLaunchConfiguration),
            typeof(CudaDim3),
            typeof(CudaDriver),
            typeof(CudaDriverCapability),
            typeof(CudaDriverModule),
            typeof(CudaDriverKernelLaunch)
        };

        foreach (Type type in rtcTypes)
        {
            IEnumerable<Type> exposedTypes = type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                .SelectMany(GetExposedTypes);
            Assert.DoesNotContain(exposedTypes, exposed =>
                exposed == typeof(IntPtr) || exposed == typeof(UIntPtr) ||
                typeof(System.Runtime.InteropServices.SafeHandle).IsAssignableFrom(exposed));
        }

        Assert.NotNull(typeof(CudaRtcArtifact).GetMethod(nameof(CudaRtcArtifact.ToArray)));
        Assert.Null(typeof(CudaRtcArtifact).GetProperty("Handle"));
        Assert.Null(typeof(CudaRtcProgram).GetProperty("Handle"));
    }

    [Fact]
    public void NativeRtcOwnerUsesOptionalLoaderCallerBuffersAndNoThrowGuards()
    {
        string native = ReadSource("native", "src", "cuda", "rtc.cpp");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-sixty-third-batch-runtime-compilation-owner.manifest.json");

        Assert.Contains("JYPPX_NVRTC_LIBRARY", native, StringComparison.Ordinal);
        Assert.Contains("LoadLibraryA", native, StringComparison.Ordinal);
        Assert.Contains("dlopen", native, StringComparison.Ordinal);
        Assert.Contains("run_rtc_noexcept", native, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_RTC_GUARD", native, StringComparison.Ordinal);
        Assert.Contains("embedded NUL", native, StringComparison.Ordinal);
        Assert.Contains("kMaximumOutputSize", native, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_rtc_program_get_log_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_rtc_program_copy_artifact_safe", header, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_RTC_OPTIONAL_DYNAMIC", manifest, StringComparison.Ordinal);
        Assert.DoesNotContain("target_link_libraries(jyppxtrtbridge PRIVATE CUDA::nvrtc", ReadSource("CMakeLists.txt"), StringComparison.Ordinal);
    }

    [Fact]
    public void SampleRunsOwnerBoundTypedKernelAndSeparatesPackageProof()
    {
        string sample = ReadSource("samples", "CudaRuntimeCompilation", "Program.cs");
        string readme = ReadSource("samples", "CudaRuntimeCompilation", "README.md");
        Assert.Contains("CudaRtcCompiler.Compile", sample, StringComparison.Ordinal);
        Assert.Contains("CudaKernelLibrary.Load", sample, StringComparison.Ordinal);
        Assert.Contains("CudaRtcResultCode.Compilation", sample, StringComparison.Ordinal);
        Assert.Contains("CudaKernelArgument.FromDeviceMemory", sample, StringComparison.Ordinal);
        Assert.Contains("CudaKernelArgument.FromInt32", sample, StringComparison.Ordinal);
        Assert.Contains("ownersDisposedBeforeSynchronize", sample, StringComparison.Ordinal);
        Assert.Contains("CudaDriverModule.Load", sample, StringComparison.Ordinal);
        Assert.Contains("CudaDriverKernelLaunch", sample, StringComparison.Ordinal);
        Assert.Contains("local-toolkit-driver-kernel-runtime-readback", sample, StringComparison.Ordinal);
        Assert.Contains("local-toolkit-kernel-runtime-readback", sample, StringComparison.Ordinal);
        Assert.Contains("not package-consumer", readme, StringComparison.Ordinal);
    }

    [Fact]
    public void CapabilityMatrixAndLocalSmokeKeepVersionAndProofBoundaries()
    {
        string smokeRunner = ReadSource("eng", "Invoke-CudaRtcLocalSmoke.ps1");
        Assert.Contains("CudaToolkitRoots", smokeRunner, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_TOOLKIT_ROOTS", smokeRunner, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_ROOT", smokeRunner, StringComparison.Ordinal);
        Assert.DoesNotContain("third_party\\nvidia", smokeRunner, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("C:\\Program Files\\NVIDIA", smokeRunner, StringComparison.OrdinalIgnoreCase);

        using JsonDocument matrix = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "capability-matrix.json"));
        JsonElement root = matrix.RootElement;
        JsonElement[] windows = root.GetProperty("windows").EnumerateArray().ToArray();
        JsonElement[] linux = root.GetProperty("linux").EnumerateArray().ToArray();
        Assert.Equal(new[] { "11.8", "12.1", "12.9", "13.2" }, windows.Select(item => item.GetProperty("toolkitVersion").GetString()));
        Assert.Equal(4, linux.Length);
        Assert.False(windows[0].GetProperty("capabilities").GetProperty("ltoIr").GetBoolean());
        Assert.True(windows[0].GetProperty("capabilities").GetProperty("deprecatedNvvm").GetBoolean());
        Assert.True(windows[2].GetProperty("capabilities").GetProperty("ltoIr").GetBoolean());
        Assert.False(windows[3].GetProperty("capabilities").GetProperty("deprecatedNvvm").GetBoolean());
        Assert.All(linux, item => Assert.Equal("unverified-local-assets-not-found", item.GetProperty("evidenceState").GetString()));
        Assert.False(root.GetProperty("loaderContract").GetProperty("coreBridgeStaticNvrtcLink").GetBoolean());

        using JsonDocument smoke = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "local-smoke.json"));
        Assert.Equal(4, smoke.RootElement.GetProperty("schemaVersion").GetInt32());
        JsonElement[] records = smoke.RootElement.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(4, records.Length);
        Assert.All(records, item =>
        {
            Assert.True(item.GetProperty("compileSuccess").GetBoolean());
            Assert.True(item.GetProperty("compileFailureLogCaptured").GetBoolean());
            Assert.True(item.GetProperty("ptxDeterministic").GetBoolean());
            Assert.True(item.GetProperty("cubinCapturedForSm75").GetBoolean());
        });
        Assert.All(records.Take(3), item =>
        {
            Assert.True(item.GetProperty("kernelLaunch").GetBoolean());
            Assert.True(item.GetProperty("gpuReadback").GetBoolean());
            Assert.True(item.GetProperty("correctnessProof").GetBoolean());
            Assert.True(item.GetProperty("ownersDisposedBeforeSynchronize").GetBoolean());
            Assert.Equal("local-toolkit-kernel-runtime-readback", item.GetProperty("evidenceClassification").GetString());
            Assert.Matches("^[0-9a-f]{64}$", item.GetProperty("outputSha256").GetString());
        });
        Assert.Single(records.Take(3).Select(item => item.GetProperty("outputSha256").GetString()).Distinct(StringComparer.Ordinal));
        Assert.False(records[3].GetProperty("kernelLaunch").GetBoolean());
        Assert.False(records[3].GetProperty("gpuReadback").GetBoolean());
        Assert.False(records[3].GetProperty("correctnessProof").GetBoolean());
        Assert.False(records[3].GetProperty("ownersDisposedBeforeSynchronize").GetBoolean());
        Assert.All(records, item =>
        {
            Assert.True(item.GetProperty("driverCapabilityAvailable").GetBoolean());
            Assert.True(item.GetProperty("driverVersion").GetInt32() > 0);
        });
        Assert.All(records.Take(3), item =>
        {
            Assert.True(item.GetProperty("driverLoadSucceeded").GetBoolean());
            Assert.True(item.GetProperty("driverKernelLaunch").GetBoolean());
            Assert.True(item.GetProperty("driverGpuReadback").GetBoolean());
            Assert.True(item.GetProperty("driverCorrectnessProof").GetBoolean());
            Assert.True(item.GetProperty("driverOwnersDisposedBeforeSynchronize").GetBoolean());
            Assert.Equal(item.GetProperty("outputSha256").GetString(), item.GetProperty("driverOutputSha256").GetString());
            Assert.Equal("local-toolkit-driver-kernel-runtime-readback", item.GetProperty("driverEvidenceClassification").GetString());
        });
        Assert.False(records[3].GetProperty("driverLoadSucceeded").GetBoolean());
        Assert.False(records[3].GetProperty("driverKernelLaunch").GetBoolean());
        Assert.False(records[3].GetProperty("driverCorrectnessProof").GetBoolean());
        Assert.Contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION", records[3].GetProperty("driverLoadDiagnostic").GetString(), StringComparison.Ordinal);
        Assert.True(records[2].GetProperty("loadSucceeded").GetBoolean());
        Assert.False(records[3].GetProperty("loadSucceeded").GetBoolean());
        Assert.Contains("cudaErrorUnsupportedPtxVersion", records[3].GetProperty("loadDiagnostic").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void PackageManifestsKeepRtcAsHostDependencyAndRetireVendorPackaging()
    {
        using JsonDocument runtime = JsonDocument.Parse(ReadSource("pack", "runtime", "runtime-packages.manifest.json"));
        JsonElement policy = runtime.RootElement.GetProperty("cudaRtcPackaging");
        Assert.False(policy.GetProperty("bridgeOnlyBundlesNvrtc").GetBoolean());
        Assert.True(policy.GetProperty("vendorPackagingRetired").GetBoolean());
        Assert.Equal("retired-not-packable", policy.GetProperty("fullRuntimeBundleState").GetString());
        Assert.Equal(2, policy.GetProperty("windowsAssetsByCudaVersion").GetProperty("11.8").GetArrayLength());
        Assert.Equal(2, policy.GetProperty("windowsAssetsByCudaVersion").GetProperty("13.2").GetArrayLength());
        Assert.Equal("unverified-local-assets-not-found", policy.GetProperty("linuxEvidenceState").GetString());
        Assert.Equal("not-applicable-vendor-packaging-retired", policy.GetProperty("redistributionApprovalState").GetString());
        Assert.False(policy.GetProperty("materializationAllowed").GetBoolean());
        Assert.True(policy.GetProperty("assetListsAreHostDependencyDiagnosticsOnly").GetBoolean());

        using JsonDocument split = JsonDocument.Parse(ReadSource("pack", "runtime-split", "split-runtime-packages.manifest.json"));
        JsonElement role = split.RootElement.GetProperty("cudaRtcSplitRole");
        Assert.Equal("cuda-rtc", role.GetProperty("role").GetString());
        Assert.False(role.GetProperty("bridgePackagesReferenceRole").GetBoolean());
        Assert.False(role.GetProperty("fullRuntimeCollectionsMayReferenceRole").GetBoolean());
        Assert.Equal("retired-not-packable", role.GetProperty("prototypeState").GetString());
        Assert.Equal("eng/Test-CudaRtcFullRuntimePackagingPreflight.ps1", role.GetProperty("hostDependencyAuditScript").GetString());
        Assert.False(role.GetProperty("explicitPackRequestAllowed").GetBoolean());
        Assert.Equal(
            "artifacts/cuda-runtime-compilation/full-runtime-packaging-preflight.json",
            role.GetProperty("hostDependencyAuditEvidence").GetString());
        Assert.False(policy.GetProperty("licenseTextPresenceIsRedistributionApproval").GetBoolean());
    }

    [Fact]
    public void FullRuntimeRtcPackagingPreflightVerifiesLocalAssetsAndKeepsMaterializationBlocked()
    {
        string preflight = ReadSource("eng", "Test-CudaRtcFullRuntimePackagingPreflight.ps1");
        Assert.Contains("runtimeSizeBytes", preflight, StringComparison.Ordinal);
        Assert.Contains("builtinsSha256", preflight, StringComparison.Ordinal);
        Assert.Contains("licenseTextPresent", preflight, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("RequireMaterializationReady", preflight, StringComparison.Ordinal);
        Assert.Contains("performsPackaging = $false", preflight, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", preflight, StringComparison.Ordinal);

        string splitPack = ReadSource("eng", "Invoke-LocalSplitRuntimePackage.ps1");
        Assert.Contains("$retiredRoleRequests", splitPack, StringComparison.Ordinal);
        Assert.Contains("Only the 'bridge' split package role is allowed", splitPack, StringComparison.Ordinal);
        Assert.Contains("CUDA RTC, collection, meta, and full-runtime packages are retired", splitPack, StringComparison.Ordinal);
        Assert.DoesNotContain("-RequireMaterializationReady", splitPack, StringComparison.Ordinal);

        using JsonDocument evidence = JsonDocument.Parse(ReadSource(
            "artifacts", "cuda-runtime-compilation", "full-runtime-packaging-preflight.json"));
        JsonElement root = evidence.RootElement;
        Assert.Equal("local-full-runtime-packaging-preflight", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("performsDownload").GetBoolean());
        Assert.False(root.GetProperty("performsAssetCopy").GetBoolean());
        Assert.False(root.GetProperty("performsPackaging").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());

        JsonElement summary = root.GetProperty("summary");
        Assert.Equal(18, summary.GetProperty("runtimeKeyCount").GetInt32());
        Assert.Equal(6, summary.GetProperty("windowsRuntimeKeyCount").GetInt32());
        Assert.Equal(12, summary.GetProperty("linuxRuntimeKeyCount").GetInt32());
        Assert.Equal(4, summary.GetProperty("windowsToolkitVersionCount").GetInt32());
        Assert.Equal(4, summary.GetProperty("windowsAssetPairReadyCount").GetInt32());
        Assert.Equal(4, summary.GetProperty("windowsLicenseTextPresentCount").GetInt32());
        Assert.True(summary.GetProperty("windowsUniqueAssetBytes").GetInt64() > 0);
        Assert.True(summary.GetProperty("windowsUniqueAssetMiB").GetDouble() > 0);
        Assert.True(summary.GetProperty("windowsAssetStagingReady").GetBoolean());
        Assert.Equal(4, summary.GetProperty("linuxToolkitVersionCount").GetInt32());
        Assert.Equal(0, summary.GetProperty("linuxAssetReadyCount").GetInt32());
        Assert.False(summary.GetProperty("linuxAssetStagingReady").GetBoolean());
        Assert.False(summary.GetProperty("redistributionApproved").GetBoolean());
        Assert.False(summary.GetProperty("packageHostSizeReviewApproved").GetBoolean());
        Assert.False(summary.GetProperty("roleMaterialized").GetBoolean());
        Assert.Equal(0, summary.GetProperty("structuralFindingCount").GetInt32());
        Assert.Equal(4, summary.GetProperty("materializationBlockerCount").GetInt32());
        Assert.False(summary.GetProperty("canMaterializeFullRuntimeCudaRtcRole").GetBoolean());
        Assert.False(summary.GetProperty("canPublish").GetBoolean());

        JsonElement[] windows = root.GetProperty("windows").EnumerateArray().ToArray();
        Assert.Equal(
            new[] { "11.8", "12.1", "12.9", "13.2" },
            windows.Select(static item => item.GetProperty("toolkitVersion").GetString()!).ToArray());
        Assert.All(windows, item =>
        {
            Assert.True(item.GetProperty("pairMatchesCapabilityMatrix").GetBoolean());
            Assert.True(item.GetProperty("assetIntegrityReady").GetBoolean());
            Assert.True(item.GetProperty("licenseTextPresent").GetBoolean());
            Assert.True(item.GetProperty("localAssetStagingReady").GetBoolean());
            Assert.False(item.GetProperty("packageMaterializationReady").GetBoolean());
            JsonElement[] assets = item.GetProperty("assets").EnumerateArray().ToArray();
            Assert.Equal(2, assets.Length);
            Assert.All(assets, asset =>
            {
                Assert.True(asset.GetProperty("exists").GetBoolean());
                Assert.True(asset.GetProperty("sizeMatches").GetBoolean());
                Assert.True(asset.GetProperty("hashMatches").GetBoolean());
                Assert.True(asset.GetProperty("integrityReady").GetBoolean());
                Assert.Matches("^[0-9a-f]{64}$", asset.GetProperty("actualSha256").GetString());
            });
        });

        Assert.All(root.GetProperty("linux").EnumerateArray(), item =>
        {
            Assert.Equal("unverified-local-assets-not-found", item.GetProperty("capabilityEvidenceState").GetString());
            Assert.False(item.GetProperty("assetsVerified").GetBoolean());
            Assert.False(item.GetProperty("packageMaterializationReady").GetBoolean());
        });
        Assert.Empty(root.GetProperty("structuralFindings").EnumerateArray());
        string[] blockers = root.GetProperty("materializationBlockers").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(
            new[]
            {
                "redistribution-approval-pending", "package-host-size-review-pending",
                "linux-assets-unverified", "cuda-rtc-role-not-materialized"
            },
            blockers);
    }

    [Fact]
    public void NativeAbiEvidenceMatchesRtcManifestHeaderAndPeExports()
    {
        using JsonDocument abi = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "native-abi-surface.json"));
        JsonElement root = abi.RootElement;
        Assert.True(root.GetProperty("passed").GetBoolean());
        Assert.Equal(12, root.GetProperty("manifestEntryPointCount").GetInt32());
        Assert.Equal(12, root.GetProperty("declaredEntryPointCount").GetInt32());
        Assert.Equal(12, root.GetProperty("matchedPeExportCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingDeclarationCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingPeExportCount").GetInt32());

        using JsonDocument launchAbi = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "kernel-launch-native-abi-surface.json"));
        JsonElement launchRoot = launchAbi.RootElement;
        Assert.True(launchRoot.GetProperty("passed").GetBoolean());
        Assert.Equal(4, launchRoot.GetProperty("manifestEntryPointCount").GetInt32());
        Assert.Equal(4, launchRoot.GetProperty("declaredEntryPointCount").GetInt32());
        Assert.Equal(4, launchRoot.GetProperty("matchedPeExportCount").GetInt32());
        Assert.Equal(0, launchRoot.GetProperty("missingDeclarationCount").GetInt32());
        Assert.Equal(0, launchRoot.GetProperty("missingPeExportCount").GetInt32());

        using JsonDocument driverAbi = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "driver-native-abi-surface.json"));
        JsonElement driverRoot = driverAbi.RootElement;
        Assert.True(driverRoot.GetProperty("passed").GetBoolean());
        Assert.Equal(9, driverRoot.GetProperty("manifestEntryPointCount").GetInt32());
        Assert.Equal(9, driverRoot.GetProperty("declaredEntryPointCount").GetInt32());
        Assert.Equal(9, driverRoot.GetProperty("matchedPeExportCount").GetInt32());
        Assert.Equal(0, driverRoot.GetProperty("missingDeclarationCount").GetInt32());
        Assert.Equal(0, driverRoot.GetProperty("missingPeExportCount").GetInt32());
    }

    [Fact]
    public void TypedKernelLaunchRetainsOwnersAndRejectsRawPointers()
    {
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "kernel_library_launch.inc");
        string managed = ReadSource("src", "JYPPX.CudaSharp", "Kernels", "CudaKernelLaunch.cs");
        string argument = ReadSource("src", "JYPPX.CudaSharp", "Kernels", "CudaKernelArgument.cs");
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-sixty-fourth-batch-owner-bound-kernel-launch.manifest.json");

        Assert.Contains("cudaLibraryGetKernel", native, StringComparison.Ordinal);
        Assert.Contains("cudaLaunchKernel", native, StringComparison.Ordinal);
        Assert.Contains("cudaEventRecord", native, StringComparison.Ordinal);
        Assert.Contains("cudaEventSynchronize", native, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_KERNEL_LAUNCH_GUARD", native, StringComparison.Ordinal);
        Assert.Contains("SafeCudaHandleLease.Create", managed, StringComparison.Ordinal);
        Assert.Contains("DangerousAddRef", ReadSource("src", "JYPPX.CudaSharp", "Internal", "Handles", "SafeCudaHandleLease.cs"), StringComparison.Ordinal);
        Assert.Contains("FromDeviceMemory", argument, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", managed + argument, StringComparison.Ordinal);
        Assert.Contains("borrowed-call-only", manifest, StringComparison.Ordinal);
    }

    [Fact]
    public void TypedKernelLaunchValidatesManagedInputsBeforeNativeInterop()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaDim3(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaDim3(1, 0));
        Assert.Throws<InvalidOperationException>(() => new CudaKernelLaunchConfiguration(default, new CudaDim3(1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaKernelLaunchConfiguration(new CudaDim3(1), new CudaDim3(1), -1));

        CudaKernelLibrary library = (CudaKernelLibrary)RuntimeHelpers.GetUninitializedObject(typeof(CudaKernelLibrary));
        CudaStream stream = (CudaStream)RuntimeHelpers.GetUninitializedObject(typeof(CudaStream));
        var configuration = new CudaKernelLaunchConfiguration(new CudaDim3(1), new CudaDim3(32));
        CudaKernelArgument scalar = CudaKernelArgument.FromInt32(1);

        Assert.Throws<ArgumentException>(() => library.Launch("bad\0name", configuration, stream, scalar));
        Assert.Throws<ArgumentNullException>(() => library.Launch("kernel", configuration, null!, scalar));
        Assert.Throws<ArgumentNullException>(() => library.Launch("kernel", configuration, stream, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => library.Launch(
            "kernel",
            configuration,
            stream,
            Enumerable.Repeat(scalar, 257).ToArray()));
        Assert.Throws<ArgumentException>(() => library.Launch(
            "kernel",
            configuration,
            stream,
            new CudaKernelArgument[] { null! }));

        Type handleType = typeof(CudaMemory).Assembly.GetType("JYPPX.CudaSharp.Internal.Handles.SafeCudaMemoryHandle", throwOnError: true)!;
        using IDisposable closedHandle = (IDisposable)Activator.CreateInstance(handleType, nonPublic: true)!;
        closedHandle.Dispose();
        CudaMemory disposedMemory = (CudaMemory)RuntimeHelpers.GetUninitializedObject(typeof(CudaMemory));
        typeof(CudaMemory).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(disposedMemory, closedHandle);
        Assert.Throws<ObjectDisposedException>(() => CudaKernelArgument.FromDeviceMemory(disposedMemory));
    }

    [Fact]
    public void TypedKernelScalarFactoriesExposeStableCopiedMetadata()
    {
        (CudaKernelArgument Argument, CudaKernelScalarType Type, int Size)[] cases =
        {
            (CudaKernelArgument.FromBoolean(true), CudaKernelScalarType.Boolean, 1),
            (CudaKernelArgument.FromByte(byte.MaxValue), CudaKernelScalarType.Byte, 1),
            (CudaKernelArgument.FromSByte(sbyte.MinValue), CudaKernelScalarType.SByte, 1),
            (CudaKernelArgument.FromInt16(short.MinValue), CudaKernelScalarType.Int16, 2),
            (CudaKernelArgument.FromUInt16(ushort.MaxValue), CudaKernelScalarType.UInt16, 2),
            (CudaKernelArgument.FromInt32(int.MinValue), CudaKernelScalarType.Int32, 4),
            (CudaKernelArgument.FromUInt32(uint.MaxValue), CudaKernelScalarType.UInt32, 4),
            (CudaKernelArgument.FromInt64(long.MinValue), CudaKernelScalarType.Int64, 8),
            (CudaKernelArgument.FromUInt64(ulong.MaxValue), CudaKernelScalarType.UInt64, 8),
            (CudaKernelArgument.FromSingle(float.NaN), CudaKernelScalarType.Single, 4),
            (CudaKernelArgument.FromDouble(double.PositiveInfinity), CudaKernelScalarType.Double, 8)
        };

        Assert.All(cases, item =>
        {
            Assert.Equal(CudaKernelArgumentKind.Scalar, item.Argument.Kind);
            Assert.Equal(item.Type, item.Argument.ScalarType);
            Assert.Equal(item.Size, item.Argument.ScalarSizeInBytes);
            Assert.Equal(0, item.Argument.MemoryOffset);
        });
    }

    [Fact]
    public void DriverCapabilityMatrixKeepsDynamicLoaderAndLinuxBoundaries()
    {
        using JsonDocument matrix = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "driver-capability-matrix.json"));
        JsonElement root = matrix.RootElement;
        JsonElement[] windows = root.GetProperty("windows").EnumerateArray().ToArray();
        JsonElement[] linux = root.GetProperty("linux").EnumerateArray().ToArray();
        Assert.Equal(new[] { "11.8", "12.1", "12.9", "13.2" }, windows.Select(item => item.GetProperty("toolkitVersion").GetString()));
        Assert.All(windows, item =>
        {
            Assert.Equal("local-header-import-lib-driver-export-verified", item.GetProperty("evidenceState").GetString());
            JsonElement capabilities = item.GetProperty("capabilities");
            Assert.True(capabilities.GetProperty("moduleLoad").GetBoolean());
            Assert.True(capabilities.GetProperty("functionLookup").GetBoolean());
            Assert.True(capabilities.GetProperty("typedLaunch").GetBoolean());
            Assert.True(capabilities.GetProperty("failureCleanupSynchronization").GetBoolean());
            Assert.True(capabilities.GetProperty("contextInterop").GetBoolean());
            Assert.True(capabilities.GetProperty("completionEvent").GetBoolean());
            Assert.True(item.GetProperty("symbols").GetProperty("cuStreamSynchronize").GetProperty("driverExported").GetBoolean());
        });
        Assert.All(linux, item => Assert.Equal("unverified-local-assets-not-found", item.GetProperty("evidenceState").GetString()));
        JsonElement loader = root.GetProperty("loaderContract");
        Assert.Equal("optional-dynamic", loader.GetProperty("dependencyMode").GetString());
        Assert.Equal("JYPPX_CUDA_DRIVER_LIBRARY", loader.GetProperty("exactOverrideEnvironmentVariable").GetString());
        Assert.False(loader.GetProperty("coreBridgeStaticDriverLink").GetBoolean());
    }

    [Fact]
    public void DriverModuleOwnerRetainsContextAndKeepsBorrowedFunctionInsideBridge()
    {
        string native = ReadSource("native", "src", "cuda", "driver.cpp");
        string cmake = ReadSource("CMakeLists.txt");
        string managed = ReadSource("src", "JYPPX.CudaSharp", "Drivers", "CudaDriverKernelLaunch.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Drivers", "CudaDriverModule.cs");
        Assert.Contains("JYPPX_CUDA_DRIVER_LIBRARY", native, StringComparison.Ordinal);
        Assert.Contains("LoadLibraryA", native, StringComparison.Ordinal);
        Assert.Contains("dlopen", native, StringComparison.Ordinal);
        Assert.Contains("cuDevicePrimaryCtxRetain", native, StringComparison.Ordinal);
        Assert.Contains("cuDevicePrimaryCtxRelease", native, StringComparison.Ordinal);
        Assert.Contains("cuCtxPushCurrent_v2", native, StringComparison.Ordinal);
        Assert.Contains("cuModuleLoadDataEx", native, StringComparison.Ordinal);
        Assert.Contains("cuModuleGetFunction", native, StringComparison.Ordinal);
        Assert.Contains("cuLaunchKernel", native, StringComparison.Ordinal);
        Assert.Contains("cuStreamSynchronize", native, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_DRIVER_GUARD", native, StringComparison.Ordinal);
        Assert.Contains("std::numeric_limits<unsigned int>::max", native, StringComparison.Ordinal);
        Assert.True(
            native.IndexOf("owner->retained_code.assign", StringComparison.Ordinal) <
            native.IndexOf("api.primary_ctx_retain", StringComparison.Ordinal));
        Assert.Contains("SafeCudaHandleLease.Create", managed, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_driver_compile_probe", cmake, StringComparison.Ordinal);
        Assert.DoesNotContain("CUDA::cuda_driver", cmake, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", managed, StringComparison.Ordinal);

        Assert.Throws<ArgumentNullException>(() => CudaDriverModule.Load(null!));
        Assert.Throws<ArgumentException>(() => CudaDriverModule.Load(Array.Empty<byte>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaDriverModule.Load(new byte[] { 0 }, -1));
        CudaDriverModule module = (CudaDriverModule)RuntimeHelpers.GetUninitializedObject(typeof(CudaDriverModule));
        Assert.Throws<ArgumentException>(() => module.Launch("bad\0name", default, null!, Array.Empty<CudaKernelArgument>()));
    }

    [Fact]
    public void BridgePackageConsumerUsesLocalOnlyPackageReferencesAndKeepsProofBoundaries()
    {
        string script = ReadSource("eng", "Test-CudaRtcBridgePackageConsumer.ps1");
        Assert.Contains("consumer-workspaces\\crtc", script, StringComparison.Ordinal);
        Assert.Contains("<PackageReference", script, StringComparison.Ordinal);
        Assert.Contains("$projectText -notmatch '<ProjectReference'", script, StringComparison.Ordinal);
        Assert.Contains("<clear />", script, StringComparison.Ordinal);
        Assert.Contains("JYPPX_NATIVE_BRIDGE_PATH", script, StringComparison.Ordinal);
        Assert.Contains("canPromotePublicPackageProof = $false", script, StringComparison.Ordinal);

        using JsonDocument evidence = JsonDocument.Parse(ReadSource(
            "artifacts", "cuda-runtime-compilation", "bridge-package-consumer.json"));
        JsonElement root = evidence.RootElement;
        Assert.Equal("local-feed-clean-package-consumer-candidate", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("performsDownload").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPromotePublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("canPromotePostPublishProof").GetBoolean());

        JsonElement packages = root.GetProperty("packages");
        Assert.False(packages.GetProperty("containsNvrtc").GetBoolean());
        Assert.False(packages.GetProperty("containsNvrtcBuiltins").GetBoolean());
        Assert.True(packages.GetProperty("bridge").GetProperty("containsOnlyBridgeNativeAsset").GetBoolean());
        Assert.Equal(
            packages.GetProperty("bridge").GetProperty("packageBridgeEntrySha256").GetString(),
            packages.GetProperty("bridge").GetProperty("copiedBridgeSha256").GetString());

        JsonElement consumer = root.GetProperty("consumer");
        Assert.True(consumer.GetProperty("rootOutsideRepository").GetBoolean());
        Assert.True(consumer.GetProperty("cleanWorkspaceCreated").GetBoolean());
        Assert.True(consumer.GetProperty("usesPackageReferenceOnly").GetBoolean());
        Assert.False(consumer.GetProperty("usesProjectReference").GetBoolean());
        Assert.True(consumer.GetProperty("nugetSourcesCleared").GetBoolean());
        Assert.True(consumer.GetProperty("localFeedOnly").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgeEnvironmentOverrideUsed").GetBoolean());
        Assert.True(consumer.GetProperty("restoreSucceeded").GetBoolean());
        Assert.True(consumer.GetProperty("buildSucceeded").GetBoolean());

        JsonElement diagnostic = root.GetProperty("dependencyDiagnostic");
        Assert.False(diagnostic.GetProperty("rtcAvailable").GetBoolean());
        Assert.True(diagnostic.GetProperty("diagnosticPresent").GetBoolean());
        Assert.True(diagnostic.GetProperty("driverAvailable").GetBoolean());
        Assert.True(diagnostic.GetProperty("driverVersion").GetInt32() > 0);
        Assert.False(string.IsNullOrWhiteSpace(diagnostic.GetProperty("driverLoadedLibrary").GetString()));

        JsonElement runtime = root.GetProperty("runtime");
        foreach (string property in new[]
                 {
                     "rtcAvailable", "driverAvailable", "compileSucceeded", "compileFailureDiagnosticCaptured",
                     "runtimeLibraryLaunch", "runtimeLibraryReadback", "runtimeLibraryCorrectness",
                     "runtimeLibraryOwnersReleasedBeforeSynchronize", "driverLaunch", "driverReadback",
                     "driverCorrectness", "driverOwnersReleasedBeforeSynchronize", "outputHashesMatch"
                 })
        {
            Assert.True(runtime.GetProperty(property).GetBoolean(), property);
        }
        Assert.Matches("^[0-9]+\\.[0-9]+$", runtime.GetProperty("rtcVersion").GetString());
        Assert.True(runtime.GetProperty("driverVersion").GetInt32() > 0);
        Assert.False(string.IsNullOrWhiteSpace(runtime.GetProperty("rtcLoadedLibrary").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(runtime.GetProperty("driverLoadedLibrary").GetString()));
        Assert.Equal(runtime.GetProperty("runtimeOutputSha256").GetString(), runtime.GetProperty("driverOutputSha256").GetString());
        Assert.Matches("^[0-9a-f]{64}$", runtime.GetProperty("runtimeOutputSha256").GetString());
    }

    private static IEnumerable<Type> GetExposedTypes(MemberInfo member)
    {
        if (member is PropertyInfo property)
        {
            yield return property.PropertyType;
        }
        else if (member is MethodInfo method)
        {
            yield return method.ReturnType;
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                yield return parameter.ParameterType;
            }
        }
        else if (member is ConstructorInfo constructor)
        {
            foreach (ParameterInfo parameter in constructor.GetParameters())
            {
                yield return parameter.ParameterType;
            }
        }
        else if (member is FieldInfo field)
        {
            yield return field.FieldType;
        }
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
