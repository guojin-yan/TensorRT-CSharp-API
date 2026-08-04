using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.Shared.Interop;

/// <summary>
/// Shared native bridge loader used by managed assemblies that P/Invoke the bridge.
/// 供托管程序集调用的共享原生 bridge 加载器。
/// </summary>
public static class NativeBridgeLibraryLoader
{
    private static readonly object SyncRoot = new object();
    private static readonly HashSet<Assembly> InitializedAssemblies = new HashSet<Assembly>();

    /// <summary>
    /// Ensures that the native bridge loader and resolver are initialized for an assembly.
    /// 确保为指定程序集初始化原生 bridge 加载器与解析器。
    /// </summary>
    /// <param name="assembly">The assembly that issues bridge P/Invoke calls. 发起 bridge P/Invoke 调用的程序集。</param>
    public static void EnsureInitialized(Assembly assembly)
    {
        if (assembly == null)
        {
            throw new ArgumentNullException(nameof(assembly));
        }

        lock (SyncRoot)
        {
            if (InitializedAssemblies.Contains(assembly))
            {
                return;
            }

            NativeBridgePathResolver.EnsureProcessSearchPath(assembly);

#if NET5_0_OR_GREATER
            NativeLibrary.SetDllImportResolver(assembly, Resolve);
#endif
            InitializedAssemblies.Add(assembly);
        }
    }

#if NET5_0_OR_GREATER
    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, BridgeConstants.NativeBridgeLibraryName, StringComparison.Ordinal))
        {
            return IntPtr.Zero;
        }

        foreach (string candidate in NativeBridgePathResolver.EnumerateCandidatePaths(assembly))
        {
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
            {
                return handle;
            }
        }

        return IntPtr.Zero;
    }
#endif
}
