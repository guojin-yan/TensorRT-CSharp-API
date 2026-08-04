using System;
using System.Collections.Generic;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp.Internal;

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
            lastErrorCategory: (BridgeErrorCategory)value.LastErrorCategory,
            cudaToolkitAvailable: value.CudaToolkitAvailable != 0,
            tensorRtAvailable: value.TensorRtAvailable != 0);
    }

    public static BridgeCapabilityInfo ToManaged(NativeCapabilityInfo value)
    {
        return new BridgeCapabilityInfo(
            supportsTrt8Adapter: value.SupportsTrt8Adapter != 0,
            supportsTrt10Adapter: value.SupportsTrt10Adapter != 0,
            supportsTrt11Adapter: value.SupportsTrt11Adapter != 0,
            supportsTrt8RuntimeCreation: value.SupportsTrt8RuntimeCreation != 0,
            supportsTrt10RuntimeCreation: value.SupportsTrt10RuntimeCreation != 0,
            supportsTrt11RuntimeCreation: value.SupportsTrt11RuntimeCreation != 0,
            supportsTrt8BuilderCreation: value.SupportsTrt8BuilderCreation != 0,
            supportsTrt10BuilderCreation: value.SupportsTrt10BuilderCreation != 0,
            supportsTrt11BuilderCreation: value.SupportsTrt11BuilderCreation != 0,
            supportsLastErrorQuery: value.SupportsLastErrorQuery != 0,
            supportsBuildInfoQuery: value.SupportsBuildInfoQuery != 0,
            supportsRuntimeInfoQuery: value.SupportsRuntimeInfoQuery != 0);
    }

    public static TensorRtAdapterInfo ToManaged(NativeTensorRtAdapterInfo value)
    {
        return new TensorRtAdapterInfo(
            line: (TensorRtApiLine)value.Line,
            vendorDependencyAvailable: value.VendorDependencyAvailable != 0,
            runtimeCreationSupported: value.RuntimeCreationSupported != 0,
            builderCreationSupported: value.BuilderCreationSupported != 0,
            networkCreationSupported: value.NetworkCreationSupported != 0,
            engineDeserializationSupported: value.EngineDeserializationSupported != 0,
            detectedVersion: Utf8Interop.ReadString(value.DetectedVersion),
            statusMessage: Utf8Interop.ReadString(value.StatusMessage));
    }

    public static TensorRtTensorInfo ToManaged(NativeTensorRtTensorInfo value)
    {
        string name = ReadFixedUtf8(value.Name);

        int rank = value.Shape.NbDims < 0 ? 0 : Math.Min(value.Shape.NbDims, value.Shape.D.Length);
        int[] dims = new int[rank];
        Array.Copy(value.Shape.D, dims, rank);

        return new TensorRtTensorInfo(
            index: value.Index,
            name: name,
            dataType: MapDataType(value.DataType),
            ioMode: (TensorRtIOMode)value.IoMode,
            shape: new TensorRtDims(dims));
    }

    public static TensorRtParserErrorInfo ToManaged(NativeTensorRtParserErrorInfo value)
    {
        return new TensorRtParserErrorInfo(
            index: value.Index,
            code: value.Code,
            line: value.Line,
            node: value.Node,
            description: ReadFixedUtf8(value.Description),
            file: ReadFixedUtf8(value.File),
            functionName: ReadFixedUtf8(value.FunctionName),
            nodeName: ReadFixedUtf8(value.NodeName),
            nodeOperator: ReadFixedUtf8(value.NodeOperator));
    }

    public static TensorRtErrorRecord ToManaged(NativeTensorRtErrorRecordInfo value)
    {
        return new TensorRtErrorRecord(
            index: value.Index,
            code: value.Code,
            description: ReadFixedUtf8(value.Description));
    }

    public static TensorRtErrorRecorderSnapshot ToManaged(
        TensorRtApiLine line,
        NativeTensorRtErrorRecorderSnapshotInfo value,
        IReadOnlyList<TensorRtErrorRecord> records)
    {
        TensorRtInterfaceInfo interfaceInfo = new TensorRtInterfaceInfo(
            ReadFixedUtf8(value.InterfaceInfoKind),
            value.InterfaceInfoMajor,
            value.InterfaceInfoMinor);

        return new TensorRtErrorRecorderSnapshot(
            line,
            value.HasRecorder != 0,
            value.ErrorCount,
            value.HasOverflowed != 0,
            value.InterfaceInfoAvailable != 0,
            interfaceInfo,
            records ?? Array.Empty<TensorRtErrorRecord>());
    }

    public static TensorRtRuntimeCreateDiagnosticSnapshot ToManaged(NativeTensorRtRuntimeCreateDiagnosticInfo value)
    {
        return new TensorRtRuntimeCreateDiagnosticSnapshot(
            line: (TensorRtApiLine)value.Line,
            diagnosticAvailable: true,
            attempted: value.Attempted != 0,
            loggerHandlePresent: value.LoggerHandlePresent != 0,
            loggerPayloadPresent: value.LoggerPayloadPresent != 0,
            createInferRuntimeReturnedNonNull: value.CreateInferRuntimeReturnedNonNull != 0,
            createInferRuntimeReturnedNull: value.CreateInferRuntimeReturnedNull != 0,
            lastStatus: (BridgeStatusCode)value.LastStatus,
            tensorRtAvailable: value.TensorRtAvailable != 0,
            expectedMajor: value.ExpectedMajor,
            bridgeBuiltMajor: value.BridgeBuiltMajor,
            detectedVersion: ReadFixedUtf8(value.DetectedVersion),
            loggerCallbackAvailable: value.LoggerCallbackAvailable != 0,
            loggerMessageCount: value.LoggerMessageCount,
            lastLoggerSeverity: value.LastLoggerSeverity,
            lastLoggerMessage: ReadFixedUtf8(value.LastLoggerMessage),
            createRuntimePhase: ReadFixedUtf8(value.CreateRuntimePhase),
            nativeDetail: ReadFixedUtf8(value.NativeDetail),
            diagnostic: ReadFixedUtf8(value.Diagnostic));
    }

    private static TensorRtDataType MapDataType(int value)
    {
        return Enum.IsDefined(typeof(TensorRtDataType), value)
            ? (TensorRtDataType)value
            : TensorRtDataType.Unknown;
    }

    public static string ReadFixedUtf8(byte[] value)
    {
        int terminator = Array.IndexOf(value, (byte)0);
        int length = terminator >= 0 ? terminator : value.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(value, 0, length);
    }
}
