using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void ClearBuilderConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlag flag)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearBuilderConfigFlag));
        int nativeFlag = TensorRtBuilderFlagMapper.ToNativeFlag(line, flag);
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_config_clear_flag(config, nativeFlag));
    }

    public static bool SetBuilderConfigPluginsToSerialize(TensorRtApiLine line, SafeTensorRtObjectHandle config, IReadOnlyList<string> pluginLibraryPaths)
    {
        if (pluginLibraryPaths == null)
        {
            throw new ArgumentNullException(nameof(pluginLibraryPaths));
        }

        if (pluginLibraryPaths.Count == 0)
        {
            int cleared;
            BridgeStatusCode clearStatus = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out cleared),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out cleared),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out cleared),
                _ => throw UnsupportedLine()
            };

            NativeStatus.ThrowIfFailed(clearStatus);
            return cleared != 0;
        }

        Utf8Interop.Utf8StringScope[] scopes = new Utf8Interop.Utf8StringScope[pluginLibraryPaths.Count];
        IntPtr[] pointers = new IntPtr[pluginLibraryPaths.Count];
        try
        {
            for (int i = 0; i < pluginLibraryPaths.Count; i++)
            {
                if (string.IsNullOrWhiteSpace(pluginLibraryPaths[i]))
                {
                    throw new ArgumentException("Plugin library path entries must not be null or empty.", nameof(pluginLibraryPaths));
                }

                scopes[i] = Utf8Interop.ToNativeString(pluginLibraryPaths[i]);
                pointers[i] = scopes[i].Pointer;
            }

            GCHandle pinned = GCHandle.Alloc(pointers, GCHandleType.Pinned);
            try
            {
                int set;
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_plugins_to_serialize(config, pinned.AddrOfPinnedObject(), pointers.Length, out set),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_plugins_to_serialize(config, pinned.AddrOfPinnedObject(), pointers.Length, out set),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_plugins_to_serialize(config, pinned.AddrOfPinnedObject(), pointers.Length, out set),
                    _ => throw UnsupportedLine()
                };

                NativeStatus.ThrowIfFailed(status);
                return set != 0;
            }
            finally
            {
                pinned.Free();
            }
        }
        finally
        {
            for (int i = 0; i < scopes.Length; i++)
            {
                scopes[i]?.Dispose();
            }
        }
    }

}
