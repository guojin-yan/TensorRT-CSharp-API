using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace JYPPX.Shared.Interop;

/// <summary>
/// Shared native bridge loader used by managed assemblies that P/Invoke the bridge.
/// </summary>
public static class NativeBridgeLibraryLoader
{
    private static readonly object SyncRoot = new object();
    private static readonly HashSet<Assembly> InitializedAssemblies = new HashSet<Assembly>();

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
