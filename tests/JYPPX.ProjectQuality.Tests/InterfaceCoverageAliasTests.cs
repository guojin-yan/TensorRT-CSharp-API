using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class InterfaceCoverageAliasTests
{
    [Fact]
    public void TensorRtNetworkLayerAliasesCoverCompressedEntryPointTokens()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"INetworkDefinition::addElementWise\" = @(\"add-elementwise\", \"elementwise\")", script);
        Assert.Contains("\"INetworkDefinition::addSoftMax\" = @(\"add-softmax\", \"softmax\")", script);
        Assert.Contains("\"INetworkDefinition::addTopK\" = @(\"add-topk\", \"topk\")", script);
        Assert.Contains("\"INetworkDefinition::addTopKV2\" = @(\"add-topk-v2\", \"topk-v2\")", script);
        Assert.Contains("\"INetworkDefinition::addRaggedSoftMax\" = @(\"add-ragged-softmax\", \"ragged-softmax\")", script);
        Assert.Contains("\"INetworkDefinition::addNMSV2\" = @(\"add-nms\", \"nms\")", script);
        Assert.Contains("\"INetworkDefinition::addNonZeroV2\" = @(\"add-non-zero\", \"non-zero\")", script);
        Assert.Contains("\"INetworkDefinition::addParametricReLU\" = @(\"add-parametric-relu\", \"parametric-relu\")", script);
        Assert.Contains("\"INetworkDefinition::addMoE\" = @(\"add-moe\", \"moe\")", script);
    }

    [Fact]
    public void TensorRtMetadataAliasesCoverKnownFalseNegativeRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IGatherLayer::getNbElementWiseDims\" = @(\"get-nb-elementwise-dims\", \"nb-elementwise-dims\", \"get-elementwise-dims\", \"elementwise-dims\")", script);
        Assert.Contains("\"IGatherLayer::setNbElementWiseDims\" = @(\"set-nb-elementwise-dims\", \"nb-elementwise-dims\", \"set-elementwise-dims\", \"elementwise-dims\")", script);
        Assert.Contains("\"INMSLayer::getTopKBoxLimit\" = @(\"get-topk-box-limit\", \"topk-box-limit\")", script);
        Assert.Contains("\"INMSLayer::setTopKBoxLimit\" = @(\"set-topk-box-limit\", \"topk-box-limit\")", script);
        Assert.Contains("\"IOptimizationProfile::getDimensions\" = @(\"get-shape\", \"profile-shape\")", script);
        Assert.Contains("\"IOptimizationProfile::setDimensions\" = @(\"set-shape\", \"profile-shape\")", script);
        Assert.Contains("\"IMoELayer\" { $aliases.Add(\"moe-layer\") | Out-Null; $aliases.Add(\"moe\") | Out-Null }", script);
    }

    [Fact]
    public void TensorRtTimingCacheAliasesCoverSafeControlWrappers()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IHostMemory::data\" = @(\"host-memory-copy-to-buffer\", \"copy-to-buffer\")", script);
        Assert.Contains("\"ITimingCache::combine\" = @(\"timing-cache-combine\", \"combine\")", script);
        Assert.Contains("\"ITimingCache::reset\" = @(\"timing-cache-reset\", \"reset\")", script);
        Assert.Contains("\"ITimingCache::queryKeys\" = @(\"timing-cache-query-key-count\", \"timing-cache-copy-keys\", \"query-keys\")", script);
        Assert.Contains("\"ITimingCache::query\" = @(\"timing-cache-query\", \"query\")", script);
        Assert.Contains("\"ITimingCache::update\" = @(\"timing-cache-update\", \"update\")", script);
    }

    [Fact]
    public void TensorRtGlobalAliasesCoverReadonlyProbeFalseNegativeRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"Global::getNvOnnxParserVersion\" = @(\"global-get-onnx-parser-version\")", script);
        Assert.Contains("\"Global::getBuilderPluginRegistry\" = @(\"builder-capability-plugin-registry\")", script);
        Assert.Contains("function Find-ExplicitTensorRtInterfaceAliasApis", script);
    }

    [Fact]
    public void TensorRtGlobalAliasesCoverFactoryAndDeferredFalseNegativeRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"Global::createInferBuilder_INTERNAL\" = @(\"id:*builder-create\")", script);
        Assert.Contains("\"Global::createInferRuntime_INTERNAL\" = @(\"id:*runtime-create\")", script);
        Assert.Contains("\"Global::createInferRefitter_INTERNAL\" = @(\"id:*engine-create-refitter\")", script);
        Assert.Contains("\"Global::createNvOnnxParser_INTERNAL\" = @(\"id:*onnx-parser-create\")", script);
        Assert.Contains("\"Global::createNvOnnxParserRefitter_INTERNAL\" = @(\"id:*parser-refitter-create-deferred\")", script);
        Assert.Contains("\"Global::createONNXConfig\" = @(\"id:*onnx-config-create-deferred\")", script);
        Assert.Contains("\"Global::initLibNvInferPlugins\" = @(\"id:*global-init-lib-nvinfer-plugins-deferred\")", script);
        Assert.Contains("\"Global::setInternalLibraryPath\" = @(\"id:*global-set-internal-library-path-deferred\")", script);
    }

    [Fact]
    public void CudaMemoryAliasesCoverKnownFalseNegativeRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"cudaMallocHost\" = @(\"malloc-host\", \"pinned-memory-allocate\", \"host-alloc\")", script);
        Assert.Contains("\"cudaMalloc3D\" = @(\"malloc-3d\", \"pitched-memory-allocate-3d\")", script);
        Assert.Contains("\"cudaMalloc3DArray\" = @(\"malloc-3d-array\")", script);
        Assert.Contains("\"cudaMemcpy2DAsync\" = @(\"memcpy-2d-async\", \"pitched-memory-copy-async\", \"copy-2d-async\")", script);
        Assert.Contains("\"cudaMemcpy2DToArray\" = @(\"memcpy-2d-to-array\", \"copy-2d-to-array\")", script);
        Assert.Contains("\"cudaMemcpy2DFromArray\" = @(\"memcpy-2d-from-array\", \"copy-2d-from-array\")", script);
        Assert.Contains("\"cudaMemcpy2DArrayToArray\" = @(\"memcpy-2d-array-to-array\", \"copy-2d-array-to-array\")", script);
        Assert.Contains("\"cudaMemcpy3D\" = @(\"memcpy-3d\", \"pitched-memory-copy-3d\", \"copy-3d\")", script);
        Assert.Contains("\"cudaMemcpy3DAsync\" = @(\"memcpy-3d-async\", \"pitched-memory-copy-3d-async\", \"copy-3d-async\")", script);
        Assert.Contains("\"cudaMemcpy3DPeer\" = @(\"memcpy-3d-peer\")", script);
        Assert.Contains("\"cudaMemcpy3DPeerAsync\" = @(\"memcpy-3d-peer-async\")", script);
        Assert.Contains("\"cudaMemcpy3DBatchAsync\" = @(\"memcpy-3d-batch-async\")", script);
        Assert.Contains("\"cudaMemcpy3DWithAttributesAsync\" = @(\"memcpy-3d-with-attributes-async\")", script);
        Assert.Contains("\"cudaMemcpyToSymbol\" = @(\"memcpy-to-symbol\")", script);
        Assert.Contains("\"cudaMemcpyFromSymbol\" = @(\"memcpy-from-symbol\")", script);
        Assert.Contains("\"cudaMemcpyToSymbolAsync\" = @(\"memcpy-to-symbol-async\")", script);
        Assert.Contains("\"cudaMemcpyFromSymbolAsync\" = @(\"memcpy-from-symbol-async\")", script);
        Assert.Contains("\"cudaMemGetDefaultMemPool\" = @(\"mem-get-default-mem-pool\")", script);
        Assert.Contains("\"cudaMemGetMemPool\" = @(\"mem-get-mem-pool\")", script);
        Assert.Contains("\"cudaMemSetMemPool\" = @(\"mem-set-mem-pool\")", script);
        Assert.Contains("\"cudaMemPoolExportPointer\" = @(\"mem-pool-export-pointer\")", script);
        Assert.Contains("\"cudaMemPoolImportPointer\" = @(\"mem-pool-import-pointer\")", script);
    }

    [Fact]
    public void CudaDeviceAndStreamAliasesCoverBoundaryRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"cudaDeviceGetTexture1DLinearMaxWidth\" = @(\"device-get-texture-1d-linear-max-width\")", script);
        Assert.Contains("\"cudaDeviceGetGraphMemAttribute\" = @(\"device-get-graph-memory-attribute\")", script);
        Assert.Contains("\"cudaDeviceGetHostAtomicCapabilities\" = @(\"device-get-host-atomic-capabilities\")", script);
        Assert.Contains("\"cudaDeviceGetExecutionCtx\" = @(\"device-get-execution-ctx-deferred\")", script);
        Assert.Contains("\"cudaStreamAttachMemAsync\" = @(\"stream-attach-mem-async\")", script);
        Assert.Contains("\"cudaStreamGetAttribute\" = @(\"stream-get-attribute\")", script);
        Assert.Contains("\"cudaStreamGetCaptureInfo_v3\" = @(\"stream-get-capture-info-v3-deferred\")", script);
        Assert.Contains("\"cudaStreamUpdateCaptureDependencies_v2\" = @(\"stream-update-capture-dependencies-v2-deferred\")", script);
    }

    [Fact]
    public void CudaKernelLaunchAliasesCoverFunctionAndOccupancyRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"cudaFuncGetAttributes\" = @(\"func-get-attributes\")", script);
        Assert.Contains("\"cudaFuncGetParamCount\" = @(\"func-get-param-count\")", script);
        Assert.Contains("\"cudaLaunchKernelExC\" = @(\"launch-kernel-ex-c\")", script);
        Assert.Contains("\"cudaLaunchHostFunc_v2\" = @(\"launch-host-func-v2-deferred\")", script);
        Assert.Contains("\"cudaOccupancyAvailableDynamicSMemPerBlock\" = @(\"occupancy-available-dynamic-smem-per-block\")", script);
        Assert.Contains("\"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\" = @(\"occupancy-max-active-blocks-per-multiprocessor-with-flags\")", script);
    }

    [Fact]
    public void CudaGraphAliasesCoverTopologyAndBoundaryRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"cudaGraphNodeGetDependencies\" = @(\"graph-node-get-dependency-count-safe\", \"graph-node-get-dependency-safe\")", script);
        Assert.Contains("\"cudaGraphNodeGetDependentNodes\" = @(\"graph-node-get-dependent-count-safe\", \"graph-node-get-dependent-safe\")", script);
        Assert.Contains("\"cudaGraphNodeSetEnabled\" = @(\"graph-exec-node-set-enabled-safe\")", script);
        Assert.Contains("\"cudaGraphRemoveDependencies\" = @(\"graph-remove-dependency-safe\")", script);
        Assert.Contains("\"cudaGraphKernelNodeGetParams\" = @(\"graph-kernel-node-get-params-deferred\")", script);
        Assert.Contains("\"cudaGraphMemcpyNodeSetParams1D\" = @(\"graph-memcpy-node-set-params-1d-deferred\")", script);
        Assert.Contains("\"cudaGraphNodeGetParams\" = @(\"graph-node-get-params-deferred\")", script);
        Assert.Contains("\"cudaUserObjectCreate\" = @(\"user-object-create-deferred\")", script);
    }

    [Fact]
    public void CudaGraphBoundaryManifestCoversUnsafeNodeAndUserObjectRows()
    {
        string manifest = ReadCudaManifest("cuda-thirty-seventh-batch-graph-boundaries.manifest.json");

        Assert.Contains("cuda-graph-kernel-node-get-attribute-deferred", manifest);
        Assert.Contains("cuda-graph-mem-alloc-node-get-params-deferred", manifest);
        Assert.Contains("cuda-graph-memcpy-node-set-params-from-symbol-deferred", manifest);
        Assert.Contains("cuda-graph-memset-node-set-params-deferred", manifest);
        Assert.Contains("cuda-graph-node-get-containing-graph-deferred", manifest);
        Assert.Contains("cuda-graph-node-get-dependencies-v2-deferred", manifest);
        Assert.Contains("cuda-graph-remove-dependencies-v2-deferred", manifest);
        Assert.Contains("cuda-graph-retain-user-object-deferred", manifest);
        Assert.Contains("cuda-user-object-release-deferred", manifest);
    }

    [Fact]
    public void CudaOtherAliasesCoverRuntimeBoundaryRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"cudaCreateTextureObject\" = @(\"create-texture-object-deferred\")", script);
        Assert.Contains("\"cudaDestroyExternalMemory\" = @(\"destroy-external-memory-deferred\")", script);
        Assert.Contains("\"cudaSignalExternalSemaphoresAsync_v2\" = @(\"signal-external-semaphores-async-v2-deferred\")", script);
        Assert.Contains("\"cudaIpcOpenMemHandle\" = @(\"ipc-open-mem-handle-deferred\")", script);
        Assert.Contains("\"cudaGetDriverEntryPointByVersion\" = @(\"get-driver-entry-point-by-version-deferred\")", script);
        Assert.Contains("\"cudaLibraryLoadData\" = @(\"library-load-data-deferred\")", script);
        Assert.Contains("\"cudaExecutionCtxStreamCreate\" = @(\"execution-ctx-stream-create-deferred\")", script);
        Assert.Contains("\"cudaLogsRegisterCallback\" = @(\"logs-register-callback-deferred\")", script);
        Assert.Contains("\"cudaMemRangeGetAttributes\" = @(\"mem-range-get-attributes-deferred\")", script);
    }

    [Fact]
    public void CudaOtherBoundaryManifestCoversRemainingRuntimeRows()
    {
        string manifest = ReadCudaManifest("cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        Assert.Contains("cuda-create-surface-object-deferred", manifest);
        Assert.Contains("cuda-external-memory-get-mapped-mipmapped-array-deferred", manifest);
        Assert.Contains("cuda-wait-external-semaphores-async-ptsz-deferred", manifest);
        Assert.Contains("cuda-ipc-get-event-handle-deferred", manifest);
        Assert.Contains("cuda-get-symbol-address-deferred", manifest);
        Assert.Contains("cuda-library-get-unified-function-deferred", manifest);
        Assert.Contains("cuda-dev-sm-resource-split-by-count-deferred", manifest);
        Assert.Contains("cuda-green-ctx-create-deferred", manifest);
        Assert.Contains("cuda-mem-discard-and-prefetch-batch-async-deferred", manifest);
    }

    [Fact]
    public void TensorRtDeferredManifestCoversCallbackAllocatorAndPluginBoundaryRows()
    {
        string manifest = ReadTensorRt11Manifest("trt11-forty-fifth-batch-callback-deferred.manifest.json");

        Assert.Contains("trt11-error-recorder-get-error-code-deferred", manifest);
        Assert.Contains("trt11-logger-finder-find-logger-deferred", manifest);
        Assert.Contains("trt11-profiler-report-layer-time-deferred", manifest);
        Assert.Contains("trt11-debug-listener-process-debug-tensor-deferred", manifest);
        Assert.Contains("trt11-dimension-expr-is-constant-deferred", manifest);
        Assert.Contains("trt11-gpu-allocator-allocate-deferred", manifest);
        Assert.Contains("trt11-gpu-async-allocator-allocate-async-deferred", manifest);
        Assert.Contains("trt11-output-allocator-reallocate-output-deferred", manifest);
        Assert.Contains("trt11-progress-monitor-step-complete-deferred", manifest);
        Assert.Contains("trt11-stream-reader-v2-seek-deferred", manifest);
        Assert.Contains("trt11-versioned-interface-get-api-language-deferred", manifest);
        Assert.Contains("trt11-network-add-plugin-v3-deferred", manifest);
        Assert.Contains("trt11-plugin-v2-layer-get-plugin-deferred", manifest);
    }

    [Fact]
    public void TensorRtPluginDeferredManifestCoversRemainingPluginBoundaryRows()
    {
        string manifest = ReadTensorRt11Manifest("trt11-forty-sixth-batch-plugin-deferred.manifest.json");

        Assert.Contains("trt11-plugin-registry-load-library-deferred", manifest);
        Assert.Contains("trt11-plugin-registry-get-all-creators-recursive-deferred", manifest);
        Assert.Contains("trt11-plugin-resource-clone-deferred", manifest);
        Assert.Contains("trt11-plugin-resource-context-get-gpu-allocator-deferred", manifest);
        Assert.Contains("trt11-plugin-v2-enqueue-deferred", manifest);
        Assert.Contains("trt11-plugin-v2-dynamic-ext-get-output-dimensions-deferred", manifest);
        Assert.Contains("trt11-plugin-v2-io-ext-supports-format-combination-deferred", manifest);
        Assert.Contains("trt11-plugin-v3-get-capability-interface-deferred", manifest);
        Assert.Contains("trt11-plugin-v3-one-build-get-output-shapes-deferred", manifest);
        Assert.Contains("trt11-plugin-v3-one-core-get-plugin-name-deferred", manifest);
    }

    [Fact]
    public void TensorRtPluginAliasesCoverCreatorMetadataInventoryFalseNegativeRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IPluginCreatorV3One::getPluginName\" = @(\"id:*builder-capability-plugin-creator-get-name\", \"id:*builder-plugin-creator-get-name\", \"id:*global-plugin-creator-get-name\")", script);
        Assert.Contains("\"IPluginCreatorV3One::getPluginVersion\" = @(\"id:*builder-capability-plugin-creator-get-version\", \"id:*builder-plugin-creator-get-version\", \"id:*global-plugin-creator-get-version\")", script);
        Assert.Contains("\"IPluginCreatorV3One::getPluginNamespace\" = @(\"id:*builder-capability-plugin-creator-get-namespace\", \"id:*builder-plugin-creator-get-namespace\", \"id:*global-plugin-creator-get-namespace\")", script);
        Assert.Contains("\"IPluginCreatorV3One::getFieldNames\" = @(\"id:*builder-capability-plugin-creator-get-field-count\", \"id:*builder-capability-plugin-creator-get-field-name\", \"id:*builder-capability-plugin-creator-get-field-metadata\", \"id:*builder-plugin-creator-get-field-count\", \"id:*builder-plugin-creator-get-field-name\", \"id:*builder-plugin-creator-get-field-metadata\", \"id:*global-plugin-creator-get-field-count\", \"id:*global-plugin-creator-get-field-name\", \"id:*global-plugin-creator-get-field-metadata\")", script);
        Assert.Contains("\"IPluginCreatorV3One::getInterfaceInfo\" = @(\"id:*builder-capability-plugin-creator-get-interface-info\", \"id:*builder-plugin-creator-get-interface-info\", \"id:*global-plugin-creator-get-interface-info\")", script);
        Assert.Contains("\"IPluginRegistry::getPluginCreator\" = @(\"id:*plugin-creator-lookup\", \"id:*plugin-creator-lookup-get-interface-info\", \"id:*plugin-creator-lookup-get-field-count\", \"id:*plugin-creator-lookup-get-field-name\", \"id:*plugin-creator-lookup-get-field-metadata\")", script);
        Assert.Contains("\"IPluginRegistry::getAllCreators\" = @(\"id:*plugin-registry-get-creator-count\", \"id:*plugin-creator-get-name\", \"id:*plugin-creator-get-version\", \"id:*plugin-creator-get-namespace\", \"id:*plugin-creator-get-interface-info\", \"id:*plugin-creator-get-field-count\")", script);
    }

    [Fact]
    public void TensorRt10PluginCreatorAliasesCoverExistingRegistryInventoryWrappers()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IPluginCreator::getPluginName\" = @(\"id:*builder-capability-plugin-creator-get-name\", \"id:*builder-plugin-creator-get-name\", \"id:*global-plugin-creator-get-name\")", script);
        Assert.Contains("\"IPluginCreator::getPluginVersion\" = @(\"id:*builder-capability-plugin-creator-get-version\", \"id:*builder-plugin-creator-get-version\", \"id:*global-plugin-creator-get-version\")", script);
        Assert.Contains("\"IPluginCreator::getPluginNamespace\" = @(\"id:*builder-capability-plugin-creator-get-namespace\", \"id:*builder-plugin-creator-get-namespace\", \"id:*global-plugin-creator-get-namespace\")", script);
        Assert.Contains("\"IPluginCreator::getFieldNames\" = @(\"id:*builder-capability-plugin-creator-get-field-count\", \"id:*builder-capability-plugin-creator-get-field-name\", \"id:*builder-capability-plugin-creator-get-field-metadata\", \"id:*builder-plugin-creator-get-field-count\", \"id:*builder-plugin-creator-get-field-name\", \"id:*builder-plugin-creator-get-field-metadata\", \"id:*global-plugin-creator-get-field-count\", \"id:*global-plugin-creator-get-field-name\", \"id:*global-plugin-creator-get-field-metadata\")", script);
    }

    [Fact]
    public void TensorRtPluginAliasesCoverBuilderSafeRegistryExistenceFalseNegativeRow()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IPluginRegistry::getBuilderSafePluginRegistry\" = @(\"id:*builder-safe-plugin-registry-exists\", \"id:*builder-capability-plugin-registry-exists\")", script);
    }

    [Fact]
    public void TensorRt10CrossVersionNetworkManifestCoversSafeLayersAndDeferredPluginBoundaries()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-first-batch-network-compat.manifest.json");

        Assert.Contains("trt10-network-add-gather-v2", manifest);
        Assert.Contains("trt10-network-add-quantize-v2", manifest);
        Assert.Contains("trt10-network-add-dequantize-v2", manifest);
        Assert.Contains("trt10-network-add-non-zero", manifest);
        Assert.Contains("trt10-network-set-weights-name", manifest);
        Assert.Contains("trt10-network-add-plugin-v2-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-layer-get-plugin-deferred", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionPluginDeferredManifestCoversUnsafePluginBoundaries()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-second-batch-plugin-deferred.manifest.json");

        Assert.Contains("trt10-plugin-registry-load-library-deferred", manifest);
        Assert.Contains("trt10-plugin-registry-get-plugin-creator-list-deferred", manifest);
        Assert.Contains("trt10-plugin-resource-context-get-gpu-allocator-deferred", manifest);
        Assert.Contains("trt10-plugin-v2-dynamic-ext-can-broadcast-input-across-batch-deferred", manifest);
        Assert.Contains("trt10-plugin-v2-ext-is-output-broadcast-across-batch-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-get-interface-info-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-build-get-output-shapes-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-core-get-plugin-name-deferred", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionTimingCacheAndHostMemoryManifestCoversSafeOtherRows()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-third-batch-timing-cache-host-memory.manifest.json");

        Assert.Contains("jyppx-trt10-timing-cache-combine", manifest);
        Assert.Contains("jyppx-trt10-timing-cache-reset", manifest);
        Assert.Contains("jyppx-trt10-timing-cache-query-key-count", manifest);
        Assert.Contains("jyppx-trt10-timing-cache-copy-keys", manifest);
        Assert.Contains("jyppx-trt10-timing-cache-query", manifest);
        Assert.Contains("jyppx-trt10-timing-cache-update", manifest);
        Assert.Contains("jyppx-trt10-host-memory-get-type", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionOtherDeferredManifestCoversUnsafeCallbackBoundaries()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-third-batch-other-deferred.manifest.json");

        Assert.Contains("trt10-algorithm-get-timing-m-sec-deferred", manifest);
        Assert.Contains("trt10-algorithm-selector-select-algorithms-deferred", manifest);
        Assert.Contains("trt10-debug-listener-process-debug-tensor-deferred", manifest);
        Assert.Contains("trt10-dimension-expr-is-constant-deferred", manifest);
        Assert.Contains("trt10-gpu-allocator-allocate-deferred", manifest);
        Assert.Contains("trt10-gpu-async-allocator-allocate-async-deferred", manifest);
        Assert.Contains("trt10-int8-calibrator-get-batch-deferred", manifest);
        Assert.Contains("trt10-output-allocator-reallocate-output-deferred", manifest);
        Assert.Contains("trt10-progress-monitor-step-complete-deferred", manifest);
        Assert.Contains("trt10-stream-reader-v2-seek-deferred", manifest);
        Assert.Contains("trt10-versioned-interface-get-api-language-deferred", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionOnnxParserAliasesCoverSafeCopyWrappers()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IParser::getSubgraphNodes\" = @(\"get-subgraph-node-count\", \"get-subgraph-node\", \"subgraph-node\", \"subgraph-nodes\")", script);
        Assert.Contains("\"IParser::getUsedVCPluginLibraries\" = @(\"get-used-vc-plugin-library-count\", \"get-used-vc-plugin-library\", \"used-vc-plugin-library\", \"used-vc-plugin-libraries\")", script);
    }

    [Fact]
    public void TensorRt10CrossVersionOnnxParserGlobalDeferredManifestCoversUnsafeParserBoundaries()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json");

        Assert.Contains("trt10-onnx-config-get-verbosity-level-deferred", manifest);
        Assert.Contains("trt10-onnx-config-reduce-verbosity-deferred", manifest);
        Assert.Contains("trt10-onnx-config-set-verbosity-level-deferred", manifest);
        Assert.Contains("trt10-parser-parse-with-weight-descriptors-deferred", manifest);
        Assert.Contains("trt10-parser-refitter-create-deferred", manifest);
        Assert.Contains("trt10-parser-refitter-clear-errors-deferred", manifest);
        Assert.Contains("trt10-parser-refitter-get-error-deferred", manifest);
        Assert.Contains("trt10-parser-refitter-get-nb-errors-deferred", manifest);
        Assert.Contains("trt10-parser-refitter-refit-from-bytes-deferred", manifest);
        Assert.Contains("trt10-parser-refitter-refit-from-file-deferred", manifest);
        Assert.Contains("trt10-onnx-config-create-deferred", manifest);
        Assert.Contains("trt10-global-init-lib-nvinfer-plugins-deferred", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionRuntimeSerializationDeferredManifestCoversUnsafeRuntimeBoundaries()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");

        Assert.Contains("trt10-plugin-v3-one-runtime-attach-to-context-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-runtime-enqueue-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-runtime-get-fields-to-serialize-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-runtime-get-interface-info-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-runtime-on-shape-change-deferred", manifest);
        Assert.Contains("trt10-plugin-v3-one-runtime-set-tactic-deferred", manifest);
        Assert.Contains("trt10-runtime-deserialize-cuda-engine-v2-deferred", manifest);
        Assert.Contains("trt10-runtime-get-logger-deferred", manifest);
        Assert.Contains("trt10-runtime-get-plugin-registry-deferred", manifest);
        Assert.Contains("trt10-runtime-load-runtime-deferred", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionOptimizationProfileV2ManifestCoversSafeShapeValueWrappers()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-sixth-batch-optimization-profile-v2.manifest.json");

        Assert.Contains("jyppx-trt10-optimization-profile-set-shape-values-v2", manifest);
        Assert.Contains("jyppx-trt10-optimization-profile-get-shape-value-count-v2", manifest);
        Assert.Contains("jyppx-trt10-optimization-profile-get-shape-values-v2", manifest);
    }

    [Fact]
    public void TensorRt10CrossVersionDiagnosticsRefitterDeferredManifestCoversUnsafeCallbackBoundaries()
    {
        string manifest = ReadTensorRt10Manifest("trt10-cross-version-sixth-batch-diagnostics-refitter-deferred.manifest.json");

        Assert.Contains("trt10-error-recorder-dec-ref-count-deferred", manifest);
        Assert.Contains("trt10-error-recorder-enum-max-deferred", manifest);
        Assert.Contains("trt10-error-recorder-get-error-code-deferred", manifest);
        Assert.Contains("trt10-error-recorder-get-error-desc-deferred", manifest);
        Assert.Contains("trt10-error-recorder-get-interface-info-deferred", manifest);
        Assert.Contains("trt10-error-recorder-get-nb-errors-deferred", manifest);
        Assert.Contains("trt10-error-recorder-has-overflowed-deferred", manifest);
        Assert.Contains("trt10-error-recorder-inc-ref-count-deferred", manifest);
        Assert.Contains("trt10-logger-finder-find-logger-deferred", manifest);
        Assert.Contains("trt10-profiler-report-layer-time-deferred", manifest);
        Assert.Contains("trt10-refitter-set-named-weights-with-location-deferred", manifest);
    }

    [Fact]
    public void TensorRtRefitterAliasesCoverDynamicRangeSnapshotRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IRefitter::getTensorsWithDynamicRange\" = @(\"id:*refitter-get-dynamic-range-tensor-count\", \"id:*refitter-get-dynamic-range-tensor-entries\")", script);
    }

    [Fact]
    public void TensorRt8CrossVersionSeventhBatchManifestCoversSafeAndDeferredRuntimeBoundaries()
    {
        string probeManifest = ReadTensorRt8Manifest("trt8-cross-version-seventh-batch-global-runtime-probe.manifest.json");
        string deferredManifest = ReadTensorRt8Manifest("trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");

        Assert.Contains("trt8-global-get-infer-lib-version", probeManifest);
        Assert.Contains("trt8-global-get-onnx-parser-version", probeManifest);
        Assert.Contains("trt8-global-has-logger", probeManifest);
        Assert.Contains("trt8-error-recorder-get-error-code-deferred", deferredManifest);
        Assert.Contains("trt8-logger-finder-find-logger-deferred", deferredManifest);
        Assert.Contains("trt8-profiler-report-layer-time-deferred", deferredManifest);
        Assert.Contains("trt8-runtime-get-logger-deferred", deferredManifest);
        Assert.Contains("trt8-runtime-get-plugin-registry-deferred", deferredManifest);
        Assert.Contains("trt8-runtime-load-runtime-deferred", deferredManifest);
        Assert.Contains("trt8-global-create-caffe-parser-deferred", deferredManifest);
        Assert.Contains("trt8-global-create-onnx-config-deferred", deferredManifest);
        Assert.Contains("trt8-global-get-builder-plugin-registry-deferred", deferredManifest);
        Assert.Contains("trt8-global-get-safe-plugin-registry-deferred", deferredManifest);
        Assert.Contains("trt8-global-shutdown-protobuf-library-deferred", deferredManifest);
        Assert.Contains("trt8-global-transpose-sub-buffers-deferred", deferredManifest);
    }

    [Fact]
    public void TensorRtDestroyAliasesCoverCommonSafeHandleLifetimeRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IRuntime::destroy\" = @(\"id:*trt-object-destroy\")", script);
        Assert.Contains("\"IRefitter::destroy\" = @(\"id:*trt-object-destroy\")", script);
        Assert.Contains("\"IParser::destroy\" = @(\"id:*trt-object-destroy\")", script);
        Assert.Contains("\"INetworkDefinition::destroy\" = @(\"id:*trt-object-destroy\")", script);
        Assert.Contains("\"IHostMemory::destroy\" = @(\"id:*trt-object-destroy\")", script);
    }

    [Fact]
    public void TensorRt8CrossVersionEighthBatchManifestCoversOnnxParserSafeRowsAndDeferredConfigRows()
    {
        string supportManifest = ReadTensorRt8Manifest("trt8-cross-version-eighth-batch-onnx-parser-support.manifest.json");
        string deferredManifest = ReadTensorRt8Manifest("trt8-cross-version-eighth-batch-onnx-config-parser-deferred.manifest.json");

        Assert.Contains("trt8-onnx-parser-get-used-vc-plugin-library-count", supportManifest);
        Assert.Contains("trt8-onnx-parser-get-used-vc-plugin-library", supportManifest);
        Assert.Contains("trt8-onnx-parser-supports-model", supportManifest);
        Assert.Contains("trt8-onnx-config-destroy-deferred", deferredManifest);
        Assert.Contains("trt8-onnx-config-get-model-file-name-deferred", deferredManifest);
        Assert.Contains("trt8-onnx-config-set-model-file-name-deferred", deferredManifest);
        Assert.Contains("trt8-onnx-config-get-verbosity-level-deferred", deferredManifest);
        Assert.Contains("trt8-onnx-config-reduce-verbosity-deferred", deferredManifest);
        Assert.Contains("trt8-onnx-config-set-verbosity-level-deferred", deferredManifest);
        Assert.Contains("trt8-parser-parse-with-weight-descriptors-deferred", deferredManifest);
    }

    [Fact]
    public void TensorRt8CrossVersionNinthBatchManifestCoversNetworkLayerSafeRowsAndDeferredRnnRows()
    {
        string safeManifest = ReadTensorRt8Manifest("trt8-cross-version-ninth-batch-network-layer-safe.manifest.json");
        string deferredManifest = ReadTensorRt8Manifest("trt8-cross-version-ninth-batch-network-layer-deferred.manifest.json");

        Assert.Contains("trt8-network-add-fully-connected", safeManifest);
        Assert.Contains("trt8-network-add-gather-v2", safeManifest);
        Assert.Contains("trt8-network-add-ragged-softmax", safeManifest);
        Assert.Contains("trt8-network-add-parametric-relu", safeManifest);
        Assert.Contains("trt8-network-add-non-zero", safeManifest);
        Assert.Contains("trt8-network-mark-output-for-shapes", safeManifest);
        Assert.Contains("trt8-network-unmark-output-for-shapes", safeManifest);
        Assert.Contains("trt8-network-set-weights-name", safeManifest);
        Assert.Contains("trt8-fully-connected-layer-set-kernel-weights", safeManifest);
        Assert.Contains("trt8-blob-name-to-tensor-find-deferred", deferredManifest);
        Assert.Contains("trt8-network-add-plugin-v2-deferred", deferredManifest);
        Assert.Contains("trt8-network-add-rnnv2-deferred", deferredManifest);
        Assert.Contains("trt8-network-set-error-recorder-deferred", deferredManifest);
        Assert.Contains("trt8-plugin-v2-layer-get-plugin-deferred", deferredManifest);
        Assert.Contains("trt8-rnnv2-layer-get-weights-for-gate-deferred", deferredManifest);
        Assert.Contains("trt8-rnnv2-layer-set-weights-for-gate-deferred", deferredManifest);
    }

    [Fact]
    public void TensorRt8CrossVersionTenthBatchManifestCoversOtherSafeRowsAndDeferredBoundaries()
    {
        string safeManifest = ReadTensorRt8Manifest("trt8-cross-version-tenth-batch-other-safe.manifest.json");
        string deferredManifest = ReadTensorRt8Manifest("trt8-cross-version-tenth-batch-other-deferred.manifest.json");

        Assert.Contains("trt8-timing-cache-combine", safeManifest);
        Assert.Contains("trt8-timing-cache-reset", safeManifest);
        Assert.Contains("trt8-host-memory-get-type", safeManifest);
        Assert.Contains("trt8-algorithm-get-algorithm-io-info-deferred", deferredManifest);
        Assert.Contains("trt8-algorithm-io-info-get-tensor-format-deferred", deferredManifest);
        Assert.Contains("trt8-caffe-parser-parse-buffers-deferred", deferredManifest);
        Assert.Contains("trt8-gpu-allocator-free-deferred", deferredManifest);
        Assert.Contains("trt8-int8-calibrator-read-calibration-cache-deferred", deferredManifest);
        Assert.Contains("trt8-output-allocator-reallocate-output-deferred", deferredManifest);
        Assert.Contains("trt8-uff-parser-get-uff-required-version-major-deferred", deferredManifest);
    }

    [Fact]
    public void TensorRt8CrossVersionEleventhBatchManifestCoversPluginBoundaryRows()
    {
        string manifest = ReadTensorRt8Manifest("trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json");

        Assert.Contains("trt8-plugin-checker-validate-deferred", manifest);
        Assert.Contains("trt8-plugin-creator-get-plugin-name-deferred", manifest);
        Assert.Contains("trt8-plugin-registry-get-builder-safe-plugin-registry-deferred", manifest);
        Assert.Contains("trt8-plugin-registry-get-plugin-creator-list-deferred", manifest);
        Assert.Contains("trt8-plugin-v2-destroy-deferred", manifest);
        Assert.Contains("trt8-plugin-v2-get-serialization-size-deferred", manifest);
        Assert.Contains("trt8-plugin-v2-dynamic-ext-return-deferred", manifest);
        Assert.Contains("trt8-plugin-v2-ext-get-tensor-rt-version-deferred", manifest);
        Assert.Contains("trt8-plugin-v2-io-ext-supports-format-combination-deferred", manifest);
    }

    [Fact]
    public void TensorRtNetworkLayerAliasesCoverMoEGenericLayerInputAndWeightsNameRows()
    {
        string script = ReadCoverageScript();

        Assert.Contains("\"IMoELayer::setInput\" = @(\"id:*layer-set-input\")", script);
        Assert.Contains("\"INetworkDefinition::setWeightsName\" = @(\"id:*network-set-weights-name\")", script);
        Assert.Contains("\"INetworkDefinition::addRNNv2\" = @(\"id:*network-add-rnnv2*\")", script);
        Assert.Contains("\"IRNNv2Layer::getWeightsForGate\" = @(\"id:*rnnv2-layer-get-weights-for-gate*\")", script);
        Assert.Contains("\"IGpuAllocator::free\" = @(\"id:*gpu-allocator-free*\")", script);
        Assert.Contains("\"IAlgorithm::getTimingMSec\" = @(\"id:*algorithm-get-timing-msec*\")", script);
    }

    private static string ReadCoverageScript()
    {
        string path = Path.Combine(RepositoryPaths.Root, "eng", "Export-InterfaceCoverageMatrix.ps1");
        return File.ReadAllText(path);
    }

    private static string ReadCudaManifest(string manifestName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "native", "manifests", "cuda", manifestName);
        return File.ReadAllText(path);
    }

    private static string ReadTensorRt11Manifest(string manifestName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "native", "manifests", "tensorrt", "v11", manifestName);
        return File.ReadAllText(path);
    }

    private static string ReadTensorRt10Manifest(string manifestName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "native", "manifests", "tensorrt", "v10", manifestName);
        return File.ReadAllText(path);
    }

    private static string ReadTensorRt8Manifest(string manifestName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "native", "manifests", "tensorrt", "v8", manifestName);
        return File.ReadAllText(path);
    }
}
