using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal;

internal static class BridgeInfoMapper
{
    public static BridgeBuildInfo ToManaged(NativeBuildInfo value)
    {
        return new BridgeBuildInfo(
            abiVersion: unchecked((int)value.AbiVersion),
            bridgeVersionMajor: unchecked((int)value.BridgeVersionMajor),
            bridgeVersionMinor: unchecked((int)value.BridgeVersionMinor),
            bridgeVersionPatch: unchecked((int)value.BridgeVersionPatch),
            bridgeName: Utf8Interop.ReadString(value.BridgeName),
            bridgeBanner: Utf8Interop.ReadString(value.BridgeBanner),
            compilerId: Utf8Interop.ReadString(value.CompilerId),
            compilerVersion: Utf8Interop.ReadString(value.CompilerVersion),
            systemName: Utf8Interop.ReadString(value.SystemName),
            systemProcessor: Utf8Interop.ReadString(value.SystemProcessor),
            buildConfiguration: Utf8Interop.ReadString(value.BuildConfiguration),
            cudaToolkitVersion: Utf8Interop.ReadString(value.CudaToolkitVersion),
            tensorRtVersion: Utf8Interop.ReadString(value.TensorRtVersion),
            hasCudaToolkit: value.HasCudaToolkit != 0,
            hasTensorRt: value.HasTensorRt != 0,
            cudaBindingsEnabled: value.CudaBindingsEnabled != 0,
            tensorRtBindingsEnabled: value.TensorRtBindingsEnabled != 0);
    }

    public static BridgeRuntimeInfo ToManaged(NativeRuntimeInfo value)
    {
        return new BridgeRuntimeInfo(
            abiVersion: unchecked((int)value.AbiVersion),
            bridgeName: Utf8Interop.ReadString(value.BridgeName),
            bridgeBanner: Utf8Interop.ReadString(value.BridgeBanner),
            lastErrorMessage: Utf8Interop.ReadString(value.LastErrorMessage),
            lastErrorCategory: value.LastErrorCategory,
            cudaToolkitAvailable: value.CudaToolkitAvailable != 0,
            tensorRtAvailable: value.TensorRtAvailable != 0);
    }
}

