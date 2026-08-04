using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateAllocatorOwnerDryRun(TensorRtApiLine line)
    {
        SafeTensorRtObjectHandle owner;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_create(out owner),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_create(out owner),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_create(out owner),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return owner;
    }

    public static NativeTensorRtAllocatorOwnerDiagnosticInfo EmitAllocatorOwnerDryRunDiagnostic(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        ulong size,
        ulong alignment,
        string reason)
    {
        using Utf8Interop.Utf8StringScope reasonUtf8 = Utf8Interop.ToNativeString(reason ?? string.Empty);
        NativeTensorRtAllocatorOwnerDiagnosticInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_emit_diagnostic(owner, size, alignment, reasonUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_emit_diagnostic(owner, size, alignment, reasonUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_emit_diagnostic(owner, size, alignment, reasonUtf8.Pointer, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtAllocatorOwnerDiagnosticInfo GetAllocatorOwnerDryRunInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner)
    {
        NativeTensorRtAllocatorOwnerDiagnosticInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_get_info(owner, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_get_info(owner, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_get_info(owner, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtAllocatorOwnerStateInfo GetAllocatorOwnerDryRunState(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner)
    {
        NativeTensorRtAllocatorOwnerStateInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_get_state(owner, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_get_state(owner, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_get_state(owner, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtAllocatorOwnerStateInfo AttachAllocatorOwnerDryRunIntent(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        string targetKind)
    {
        using Utf8Interop.Utf8StringScope targetKindUtf8 = Utf8Interop.ToNativeString(targetKind ?? string.Empty);
        NativeTensorRtAllocatorOwnerStateInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_attach_intent(owner, targetKindUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_attach_intent(owner, targetKindUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_attach_intent(owner, targetKindUtf8.Pointer, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtAllocatorOwnerStateInfo DetachAllocatorOwnerDryRunIntent(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        string targetKind)
    {
        using Utf8Interop.Utf8StringScope targetKindUtf8 = Utf8Interop.ToNativeString(targetKind ?? string.Empty);
        NativeTensorRtAllocatorOwnerStateInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_detach_intent(owner, targetKindUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_detach_intent(owner, targetKindUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_detach_intent(owner, targetKindUtf8.Pointer, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtAllocatorOwnerStateInfo RecordAllocatorOwnerDryRunAllocationIntent(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        ulong size,
        ulong alignment,
        ulong streamValue)
    {
        NativeTensorRtAllocatorOwnerStateInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_ledger_record_allocation_intent(owner, size, alignment, streamValue, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_ledger_record_allocation_intent(owner, size, alignment, streamValue, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_ledger_record_allocation_intent(owner, size, alignment, streamValue, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtAllocatorOwnerStateInfo RecordAllocatorOwnerDryRunReleaseIntent(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        ulong allocationId,
        ulong streamValue)
    {
        NativeTensorRtAllocatorOwnerStateInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_allocator_owner_dry_run_ledger_record_release_intent(owner, allocationId, streamValue, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_allocator_owner_dry_run_ledger_record_release_intent(owner, allocationId, streamValue, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_allocator_owner_dry_run_ledger_record_release_intent(owner, allocationId, streamValue, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }
}
