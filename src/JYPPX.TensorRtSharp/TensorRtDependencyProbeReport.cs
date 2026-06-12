using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Identifies how a native dependency path was discovered by <see cref="TensorRtEnvironmentProbe.ProbeNativeDependencies"/>.
/// </summary>
public enum TensorRtNativeDependencySource
{
    /// <summary>
    /// The entry is a candidate path produced by the native bridge resolver.
    /// </summary>
    NativeBridgeCandidate = 0,

    /// <summary>
    /// The entry is a module already loaded in the current process.
    /// </summary>
    LoadedProcessModule = 1,

    /// <summary>
    /// The entry is a matching DLL found on the process search path.
    /// </summary>
    SearchPathCandidate = 2
}

/// <summary>
/// Describes a native bridge, TensorRT, CUDA, cuDNN, or parser DLL discovered by a dependency probe.
/// </summary>
public sealed class TensorRtNativeDependencyInfo
{
    internal TensorRtNativeDependencyInfo(
        TensorRtNativeDependencySource source,
        string name,
        string path,
        bool exists,
        string fileVersion,
        string productVersion,
        string diagnostic)
    {
        Source = source;
        Name = name ?? string.Empty;
        Path = path ?? string.Empty;
        Exists = exists;
        FileVersion = fileVersion ?? string.Empty;
        ProductVersion = productVersion ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets how this dependency path was discovered.
    /// </summary>
    public TensorRtNativeDependencySource Source { get; }

    /// <summary>
    /// Gets the module or file name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the full path when it is available.
    /// </summary>
    public string Path { get; }

    /// <summary>
    /// Gets whether the path exists at probe time.
    /// </summary>
    public bool Exists { get; }

    /// <summary>
    /// Gets the file version reported by Windows file-version metadata when available.
    /// </summary>
    public string FileVersion { get; }

    /// <summary>
    /// Gets the product version reported by Windows file-version metadata when available.
    /// </summary>
    public string ProductVersion { get; }

    /// <summary>
    /// Gets a non-throwing diagnostic collected while reading this entry.
    /// </summary>
    public string Diagnostic { get; }

    public override string ToString()
    {
        string version = !string.IsNullOrWhiteSpace(FileVersion) ? FileVersion : ProductVersion;
        return $"{Source}:{Name}:{(Exists ? "exists" : "missing")}:{(string.IsNullOrWhiteSpace(version) ? "version=n/a" : $"version={version}")}";
    }
}

/// <summary>
/// Non-throwing diagnostic report for native bridge, TensorRT, parser, CUDA, and cuDNN dependency resolution.
/// </summary>
public sealed class TensorRtDependencyProbeReport
{
    internal TensorRtDependencyProbeReport(
        TensorRtApiLine line,
        bool bridgeInitialized,
        string bridgeDiagnostic,
        IReadOnlyList<TensorRtNativeDependencyInfo> nativeBridgeCandidates,
        IReadOnlyList<TensorRtNativeDependencyInfo> loadedModules,
        IReadOnlyList<TensorRtNativeDependencyInfo> searchPathCandidates,
        IReadOnlyList<string> diagnostics)
    {
        Line = line;
        BridgeInitialized = bridgeInitialized;
        BridgeDiagnostic = bridgeDiagnostic ?? string.Empty;
        NativeBridgeCandidates = nativeBridgeCandidates ?? Array.Empty<TensorRtNativeDependencyInfo>();
        LoadedModules = loadedModules ?? Array.Empty<TensorRtNativeDependencyInfo>();
        SearchPathCandidates = searchPathCandidates ?? Array.Empty<TensorRtNativeDependencyInfo>();
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the TensorRT API line used to choose expected TensorRT DLL names.
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether the managed native bridge loader initialized successfully.
    /// </summary>
    public bool BridgeInitialized { get; }

    /// <summary>
    /// Gets the bridge loader diagnostic message.
    /// </summary>
    public string BridgeDiagnostic { get; }

    /// <summary>
    /// Gets native bridge candidate paths produced by the same resolver used for DllImport loading.
    /// </summary>
    public IReadOnlyList<TensorRtNativeDependencyInfo> NativeBridgeCandidates { get; }

    /// <summary>
    /// Gets matching modules that are already loaded in the current process.
    /// </summary>
    public IReadOnlyList<TensorRtNativeDependencyInfo> LoadedModules { get; }

    /// <summary>
    /// Gets matching dependency DLLs found on the current process search path.
    /// </summary>
    public IReadOnlyList<TensorRtNativeDependencyInfo> SearchPathCandidates { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while probing modules and directories.
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Gets the number of matching loaded modules.
    /// </summary>
    public int LoadedModuleCount => LoadedModules.Count;

    /// <summary>
    /// Gets the number of matching search-path candidates.
    /// </summary>
    public int SearchPathCandidateCount => SearchPathCandidates.Count;

    public override string ToString()
    {
        return $"{Line}:bridge={BridgeInitialized}:loaded={LoadedModuleCount}:pathCandidates={SearchPathCandidateCount}:diagnostics={Diagnostics.Count}";
    }
}
