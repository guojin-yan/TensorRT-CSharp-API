using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool TryCreateRuntime(TensorRtApiLine line, SafeTensorRtObjectHandle logger, out string message)
    {
        return TryCreateRuntimeCore(GetBindings(line), logger, out message);
    }

    public static bool TryCreateBuilder(TensorRtApiLine line, out string message)
    {
        return TryCreateBuilderCore(GetBindings(line), out message);
    }

    public static bool TryRunTrt10MinimalBuildChain(out string message)
    {
        return TryRunTrtMinimalBuildChain(Trt10Bindings, out message);
    }

    public static bool TryBuildTrt10SerializedNetworkOnly(out string message)
    {
        return TryBuildSerializedNetworkOnly(Trt10Bindings, out message);
    }

    public static bool TryRunTrt8MinimalBuildChain(out string message)
    {
        return TryRunTrtMinimalBuildChain(Trt8Bindings, out message);
    }

    public static bool TryBuildTrt8SerializedNetworkOnly(out string message)
    {
        return TryBuildSerializedNetworkOnly(Trt8Bindings, out message);
    }

    private static bool TryRunTrtMinimalBuildChain(
        TensorRtLineBindings bindings,
        out string message)
    {
        using SafeTensorRtObjectHandle logger = CreateLogger(bindings.Line);
        BridgeStatusCode status = bindings.RuntimeCreate(logger, out SafeTensorRtObjectHandle runtime);
        if (status != BridgeStatusCode.Ok)
        {
            message = GetLastErrorMessageOrFallback($"{bindings.LineName} runtime creation failed with status '{status}'.");
            return false;
        }

        using (runtime)
        {
            status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
            if (status != BridgeStatusCode.Ok)
            {
                message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder creation failed with status '{status}'.");
                return false;
            }

            using (builder)
            {
                status = bindings.ConfigCreate(builder, out SafeTensorRtObjectHandle config);
                if (status != BridgeStatusCode.Ok)
                {
                    message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder config creation failed with status '{status}'.");
                    return false;
                }

                using (config)
                {
                    status = bindings.NetworkCreate(builder, 0, out SafeTensorRtObjectHandle network);
                    if (status != BridgeStatusCode.Ok)
                    {
                        message = GetLastErrorMessageOrFallback($"{bindings.LineName} network creation failed with status '{status}'.");
                        return false;
                    }

                    using (network)
                    {
                        status = bindings.SerializedBuild(builder, network, config, out SafeTensorRtObjectHandle hostMemory);
                        if (status != BridgeStatusCode.Ok)
                        {
                            message = GetLastErrorMessageOrFallback($"{bindings.LineName} serialized network build failed with status '{status}'.");
                            return false;
                        }

                        using (hostMemory)
                        {
                            status = bindings.DeserializeHostMemory(runtime, hostMemory, out SafeTensorRtObjectHandle engine);
                            if (status != BridgeStatusCode.Ok)
                            {
                                message = GetLastErrorMessageOrFallback($"{bindings.LineName} engine deserialization failed with status '{status}'.");
                                return false;
                            }

                            using (engine)
                            {
                                status = bindings.ExecutionContextCreate(engine, out SafeTensorRtObjectHandle context);
                                if (status != BridgeStatusCode.Ok)
                                {
                                    message = GetLastErrorMessageOrFallback($"{bindings.LineName} execution context creation failed with status '{status}'.");
                                    return false;
                                }

                                using (context)
                                {
                                    message = $"{bindings.LineName} minimal build chain completed successfully.";
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static bool TryBuildSerializedNetworkOnly(
        TensorRtLineBindings bindings,
        out string message)
    {
        using SafeTensorRtObjectHandle logger = CreateLogger(bindings.Line);
        BridgeStatusCode status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
        if (status != BridgeStatusCode.Ok)
        {
            message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder creation failed with status '{status}'.");
            return false;
        }

        using (builder)
        {
            status = bindings.NetworkCreate(builder, 0, out SafeTensorRtObjectHandle network);
            if (status != BridgeStatusCode.Ok)
            {
                message = GetLastErrorMessageOrFallback($"{bindings.LineName} network creation failed with status '{status}'.");
                return false;
            }

            using (network)
            {
                status = bindings.ConfigCreate(builder, out SafeTensorRtObjectHandle config);
                if (status != BridgeStatusCode.Ok)
                {
                    message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder config creation failed with status '{status}'.");
                    return false;
                }

                using (config)
                {
                    status = bindings.SerializedBuild(builder, network, config, out SafeTensorRtObjectHandle hostMemory);
                    if (status != BridgeStatusCode.Ok)
                    {
                        message = GetLastErrorMessageOrFallback($"{bindings.LineName} serialized network build failed with status '{status}'.");
                        return false;
                    }

                    using (hostMemory)
                    {
                        message = $"{bindings.LineName} serialized network build completed successfully.";
                        return true;
                    }
                }
            }
        }
    }

}
