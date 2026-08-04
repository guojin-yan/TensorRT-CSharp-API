using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode QueryAdapterInfoDelegate(out NativeTensorRtAdapterInfo outInfo);
    private delegate BridgeStatusCode LoggerCreateDelegate(out SafeTensorRtObjectHandle outLogger);

    private delegate BridgeStatusCode RuntimeCreateDelegate(SafeTensorRtObjectHandle logger, out SafeTensorRtObjectHandle runtime);
    private delegate BridgeStatusCode BuilderCreateDelegate(SafeTensorRtObjectHandle logger, out SafeTensorRtObjectHandle builder);
    private delegate BridgeStatusCode ConfigCreateDelegate(SafeTensorRtObjectHandle builder, out SafeTensorRtObjectHandle config);
    private delegate BridgeStatusCode NetworkCreateDelegate(SafeTensorRtObjectHandle builder, uint flags, out SafeTensorRtObjectHandle network);
    private delegate BridgeStatusCode SerializedBuildDelegate(SafeTensorRtObjectHandle builder, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle config, out SafeTensorRtObjectHandle hostMemory);
    private delegate BridgeStatusCode DeserializeHostMemoryDelegate(SafeTensorRtObjectHandle runtime, SafeTensorRtObjectHandle hostMemory, out SafeTensorRtObjectHandle engine);
    private delegate BridgeStatusCode ExecutionContextCreateDelegate(SafeTensorRtObjectHandle engine, out SafeTensorRtObjectHandle context);

    private sealed class TensorRtLineBindings
    {
        public TensorRtLineBindings(
            TensorRtApiLine line,
            string lineName,
            QueryAdapterInfoDelegate queryAdapterInfo,
            LoggerCreateDelegate createLogger,
            RuntimeCreateDelegate runtimeCreate,
            BuilderCreateDelegate builderCreate,
            ConfigCreateDelegate configCreate,
            NetworkCreateDelegate networkCreate,
            SerializedBuildDelegate serializedBuild,
            DeserializeHostMemoryDelegate deserializeHostMemory,
            ExecutionContextCreateDelegate executionContextCreate)
        {
            Line = line;
            LineName = lineName;
            QueryAdapterInfo = queryAdapterInfo;
            CreateLogger = createLogger;
            RuntimeCreate = runtimeCreate;
            BuilderCreate = builderCreate;
            ConfigCreate = configCreate;
            NetworkCreate = networkCreate;
            SerializedBuild = serializedBuild;
            DeserializeHostMemory = deserializeHostMemory;
            ExecutionContextCreate = executionContextCreate;
        }

        public TensorRtApiLine Line { get; }
        public string LineName { get; }
        public QueryAdapterInfoDelegate QueryAdapterInfo { get; }
        public LoggerCreateDelegate CreateLogger { get; }
        public RuntimeCreateDelegate RuntimeCreate { get; }
        public BuilderCreateDelegate BuilderCreate { get; }
        public ConfigCreateDelegate ConfigCreate { get; }
        public NetworkCreateDelegate NetworkCreate { get; }
        public SerializedBuildDelegate SerializedBuild { get; }
        public DeserializeHostMemoryDelegate DeserializeHostMemory { get; }
        public ExecutionContextCreateDelegate ExecutionContextCreate { get; }
    }

    private static TensorRtLineBindings GetBindings(TensorRtApiLine line)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 when !IsBridgeBuiltForTensorRt11() => Trt8Bindings,
            TensorRtApiLine.TensorRt10 when !IsBridgeBuiltForTensorRt11() => Trt10Bindings,
            TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 8/10 adapter exports are not available when the bridge is built against TensorRT 11."),
            TensorRtApiLine.TensorRt11 when IsBridgeBuiltForTensorRt11() => Trt11Bindings,
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 11 adapter exports are only available when the bridge is built against TensorRT 11."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    private static bool IsBridgeBuiltForTensorRt11()
    {
        NativeBuildInfo buildInfo = GetBuildInfo();
        string version = Utf8Interop.ReadString(buildInfo.TensorRtVersion);
        return version.StartsWith("11.", StringComparison.Ordinal);
    }

}
