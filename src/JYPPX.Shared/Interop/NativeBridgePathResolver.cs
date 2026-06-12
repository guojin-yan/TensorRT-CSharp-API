using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace JYPPX.Shared.Interop;

/// <summary>
/// Resolves candidate paths for the native bridge library.
/// </summary>
public static class NativeBridgePathResolver
{
    public static void EnsureProcessSearchPath(Assembly assembly)
    {
        List<string> preferredEntries = EnumerateDependencyDirectories(assembly)
            .Where(Directory.Exists)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        HashSet<string> currentEntries = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        List<string> existingEntries = new List<string>();
        string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (string entry in currentPath.Split(new[] { Path.PathSeparator }, StringSplitOptions.RemoveEmptyEntries))
        {
            if (currentEntries.Add(entry))
            {
                existingEntries.Add(entry);
            }
        }

        List<string> updatedEntries = new List<string>();
        foreach (string candidate in preferredEntries)
        {
            if (currentEntries.Add(candidate))
            {
                updatedEntries.Add(candidate);
            }
        }

        updatedEntries.AddRange(existingEntries);
        string updatedPath = string.Join(Path.PathSeparator.ToString(), updatedEntries);
        Environment.SetEnvironmentVariable("PATH", updatedPath);
    }

    public static IEnumerable<string> EnumerateCandidatePaths(Assembly assembly)
    {
        string fileName = GetBridgeFileName();
        string? explicitPath = Environment.GetEnvironmentVariable("JYPPX_NATIVE_BRIDGE_PATH");
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            if (Directory.Exists(explicitPath))
            {
                yield return Path.Combine(explicitPath, fileName);
            }
            else
            {
                yield return explicitPath;
            }
        }

        string baseDirectory = AppContext.BaseDirectory;
        yield return Path.Combine(baseDirectory, fileName);
        yield return Path.Combine(baseDirectory, "runtimes", GetRuntimeIdentifier(), "native", fileName);

        string currentDirectory = Directory.GetCurrentDirectory();
        if (IsDevelopmentProbingEnabled())
        {
            yield return Path.Combine(currentDirectory, fileName);
            yield return Path.Combine(currentDirectory, "build-out", "bin", "Debug", fileName);
            yield return Path.Combine(currentDirectory, "build-out", "bin", "Release", fileName);
            string buildOutRoot = Path.Combine(currentDirectory, "build-out");
            if (Directory.Exists(buildOutRoot))
            {
                foreach (string presetDir in Directory.GetDirectories(buildOutRoot))
                {
                    yield return Path.Combine(presetDir, "bin", "Debug", fileName);
                    yield return Path.Combine(presetDir, "bin", "Release", fileName);
                }
            }
        }

        string assemblyDirectory = Path.GetDirectoryName(assembly.Location) ?? baseDirectory;
        yield return Path.Combine(assemblyDirectory, fileName);
        yield return Path.Combine(assemblyDirectory, "runtimes", GetRuntimeIdentifier(), "native", fileName);
    }

    public static IEnumerable<string> EnumerateDependencyDirectories(Assembly assembly)
    {
        string? explicitBridgePath = Environment.GetEnvironmentVariable("JYPPX_NATIVE_BRIDGE_PATH");
        if (!string.IsNullOrWhiteSpace(explicitBridgePath))
        {
            if (Directory.Exists(explicitBridgePath))
            {
                yield return explicitBridgePath;
            }
            else
            {
                string? explicitBridgeDirectory = Path.GetDirectoryName(explicitBridgePath);
                if (!string.IsNullOrWhiteSpace(explicitBridgeDirectory))
                {
                    yield return explicitBridgeDirectory;
                }
            }
        }

        foreach (string rootVariable in new[] { "JYPPX_TENSORRT_ROOT", "TENSORRT_ROOT", "TensorRT_ROOT" })
        {
            string? root = Environment.GetEnvironmentVariable(rootVariable);
            if (!string.IsNullOrWhiteSpace(root))
            {
                yield return Path.Combine(root, "lib");
                yield return Path.Combine(root, "bin");
            }
        }

        string? explicitCudaBin = Environment.GetEnvironmentVariable("JYPPX_CUDA_BIN");
        if (!string.IsNullOrWhiteSpace(explicitCudaBin))
        {
            yield return explicitCudaBin;
        }

        foreach (string rootVariable in new[] { "JYPPX_CUDA_ROOT", "CUDA_PATH", "CUDAToolkit_ROOT" })
        {
            string? root = Environment.GetEnvironmentVariable(rootVariable);
            if (!string.IsNullOrWhiteSpace(root))
            {
                yield return Path.Combine(root, "bin");
                yield return Path.Combine(root, "bin", "x64");
            }
        }

        foreach (string rootVariable in new[] { "JYPPX_CUDNN_ROOT", "CUDNN_ROOT", "CUDNN_PATH" })
        {
            string? root = Environment.GetEnvironmentVariable(rootVariable);
            if (!string.IsNullOrWhiteSpace(root))
            {
                yield return Path.Combine(root, "bin");
            }
        }

        string currentDirectory = Directory.GetCurrentDirectory();
        string baseDirectory = AppContext.BaseDirectory;
        yield return baseDirectory;
        yield return Path.Combine(baseDirectory, "runtimes", GetRuntimeIdentifier(), "native");

        string assemblyDirectory = Path.GetDirectoryName(assembly.Location) ?? baseDirectory;
        yield return assemblyDirectory;
        yield return Path.Combine(assemblyDirectory, "runtimes", GetRuntimeIdentifier(), "native");

        if (IsDevelopmentProbingEnabled())
        {
            yield return Path.Combine(currentDirectory, "build-out", "bin", "Debug");
            yield return Path.Combine(currentDirectory, "build-out", "bin", "Release");
            string localTensorRtBase = Path.Combine(currentDirectory, "third_party", "nvidia");
            if (Directory.Exists(localTensorRtBase))
            {
                foreach (string candidate in Directory.GetDirectories(localTensorRtBase, "TensorRT-*"))
                {
                    yield return Path.Combine(candidate, "lib");
                    yield return Path.Combine(candidate, "bin");
                }

                foreach (string candidate in Directory.GetDirectories(localTensorRtBase, "cudnn-*"))
                {
                    yield return Path.Combine(candidate, "bin");
                }
            }

            string standardCudaBase = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NVIDIA GPU Computing Toolkit", "CUDA");
            if (Directory.Exists(standardCudaBase))
            {
                foreach (string candidate in Directory.GetDirectories(standardCudaBase, "v*"))
                {
                    yield return Path.Combine(candidate, "bin");
                    yield return Path.Combine(candidate, "bin", "x64");
                }
            }
        }
    }

    public static string GetBridgeFileName()
    {
#if JYPPX_NETFRAMEWORK
        if (IsWindows())
        {
            return $"{BridgeConstants.NativeBridgeLibraryName}.dll";
        }

        if (IsLinux())
        {
            return $"lib{BridgeConstants.NativeBridgeLibraryName}.so";
        }
#else
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return $"{BridgeConstants.NativeBridgeLibraryName}.dll";
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return $"lib{BridgeConstants.NativeBridgeLibraryName}.so";
        }
#endif

        return BridgeConstants.NativeBridgeLibraryName;
    }

    private static string GetRuntimeIdentifier()
    {
#if JYPPX_NETFRAMEWORK
        string architecture = IntPtr.Size == 8 ? "x64" : "x86";
        if (IsWindows())
        {
            return $"win-{architecture}";
        }

        if (IsLinux())
        {
            return $"linux-{architecture}";
        }
#else
        string architecture = RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "arm64" : "x64";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return $"win-{architecture}";
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return $"linux-{architecture}";
        }
#endif

        return architecture;
    }

#if JYPPX_NETFRAMEWORK
    private static bool IsWindows()
    {
        PlatformID platform = Environment.OSVersion.Platform;
        return platform == PlatformID.Win32NT
            || platform == PlatformID.Win32S
            || platform == PlatformID.Win32Windows
            || platform == PlatformID.WinCE;
    }

    private static bool IsLinux()
    {
        PlatformID platform = Environment.OSVersion.Platform;
        return platform == PlatformID.Unix;
    }
#endif

    private static bool IsDevelopmentProbingEnabled()
    {
        string? value = Environment.GetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING");
        return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
    }
}
