using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal static class TensorRtOwnerScopedVersionedMetadataQuery
{
    public static bool TryGet(
        TensorRtApiLine line,
        Func<TensorRtVersionedInterfaceMetadata> query,
        out TensorRtVersionedInterfaceMetadata metadata,
        out string diagnostic)
    {
        try
        {
            metadata = query();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (
            exception is BridgeProbeException ||
            exception is NotSupportedException ||
            exception is InvalidOperationException)
        {
            metadata = new TensorRtVersionedInterfaceMetadata(
                line,
                new TensorRtInterfaceInfo(string.Empty, 0, 0),
                TensorRtApiLanguage.Unknown);
            diagnostic = exception.Message;
            return false;
        }
    }
}

public sealed partial class TensorRtRuntime
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetRuntimeErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtRefitter
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetRefitterErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtEngine
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetEngineErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtExecutionContext
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetExecutionContextErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);

    /// <summary>Tries to copy metadata from a named output allocator. 尝试复制指定 output allocator 的元数据。</summary>
    public bool TryGetOutputAllocatorVersionedMetadata(string tensorName, out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetOutputAllocatorVersionedMetadata(tensorName, out metadata, out _);

    /// <summary>Tries to copy metadata from a named output allocator. 尝试复制指定 output allocator 的元数据。</summary>
    public bool TryGetOutputAllocatorVersionedMetadata(
        string tensorName,
        out TensorRtVersionedInterfaceMetadata metadata,
        out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetExecutionContextOutputAllocatorVersionedMetadata(Line, _handle, tensorName),
            out metadata,
            out diagnostic);

    /// <summary>Tries to copy metadata from the temporary-storage allocator. 尝试复制 temporary-storage allocator 元数据。</summary>
    public bool TryGetTemporaryStorageAllocatorVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetTemporaryStorageAllocatorVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the temporary-storage allocator. 尝试复制 temporary-storage allocator 元数据。</summary>
    public bool TryGetTemporaryStorageAllocatorVersionedMetadata(
        out TensorRtVersionedInterfaceMetadata metadata,
        out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetExecutionContextTemporaryStorageAllocatorVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);

    /// <summary>Tries to copy metadata from the attached debug listener. 尝试复制已附加 debug listener 的元数据。</summary>
    public bool TryGetDebugListenerVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetDebugListenerVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached debug listener. 尝试复制已附加 debug listener 的元数据。</summary>
    public bool TryGetDebugListenerVersionedMetadata(
        out TensorRtVersionedInterfaceMetadata metadata,
        out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetExecutionContextDebugListenerVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtBuilder
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetBuilderErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetNetworkErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtEngineInspector
{
    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetErrorRecorderVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached error recorder. 尝试复制已附加 error recorder 的元数据。</summary>
    public bool TryGetErrorRecorderVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata, out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetEngineInspectorErrorRecorderVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}

public sealed partial class TensorRtBuilderConfig
{
    /// <summary>Tries to copy metadata from the attached progress monitor. 尝试复制已附加 progress monitor 的元数据。</summary>
    public bool TryGetProgressMonitorVersionedMetadata(out TensorRtVersionedInterfaceMetadata metadata) =>
        TryGetProgressMonitorVersionedMetadata(out metadata, out _);

    /// <summary>Tries to copy metadata from the attached progress monitor. 尝试复制已附加 progress monitor 的元数据。</summary>
    public bool TryGetProgressMonitorVersionedMetadata(
        out TensorRtVersionedInterfaceMetadata metadata,
        out string diagnostic) =>
        TensorRtOwnerScopedVersionedMetadataQuery.TryGet(
            Line,
            () => NativeBridgeApi.GetBuilderConfigProgressMonitorVersionedMetadata(Line, _handle),
            out metadata,
            out diagnostic);
}
