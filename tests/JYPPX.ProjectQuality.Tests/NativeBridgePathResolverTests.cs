using System.Reflection;
using JYPPX.Shared.Interop;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class NativeBridgePathResolverTests
{
    [Fact]
    public void ExistingPreferredDependencyIsMovedAheadOfNewFallbackCandidates()
    {
        string? previousPath = Environment.GetEnvironmentVariable("PATH");
        string? previousCudaRoot = Environment.GetEnvironmentVariable("JYPPX_CUDA_ROOT");
        string? previousCudnnRoot = Environment.GetEnvironmentVariable("JYPPX_CUDNN_ROOT");
        string? previousDevelopmentProbing = Environment.GetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING");
        string testRoot = Path.Combine(RepositoryPaths.Root, "build-out", "path-resolver-test", Guid.NewGuid().ToString("N"));
        string selectedCudaRoot = Path.Combine(testRoot, "cuda-selected");
        string selectedCudaBin = Path.Combine(selectedCudaRoot, "bin");
        string fallbackCudnnRoot = Path.Combine(testRoot, "cudnn-fallback");
        string fallbackCudnnBin = Path.Combine(fallbackCudnnRoot, "bin");
        string unrelatedDirectory = Path.Combine(testRoot, "unrelated");

        try
        {
            Directory.CreateDirectory(selectedCudaBin);
            Directory.CreateDirectory(fallbackCudnnBin);
            Directory.CreateDirectory(unrelatedDirectory);
            Environment.SetEnvironmentVariable(
                "PATH",
                string.Join(Path.PathSeparator, selectedCudaBin + Path.DirectorySeparatorChar, unrelatedDirectory));
            Environment.SetEnvironmentVariable("JYPPX_CUDA_ROOT", selectedCudaRoot);
            Environment.SetEnvironmentVariable("JYPPX_CUDNN_ROOT", fallbackCudnnRoot);
            Environment.SetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING", null);

            NativeBridgePathResolver.EnsureProcessSearchPath(Assembly.GetExecutingAssembly());

            string[] entries = (Environment.GetEnvironmentVariable("PATH") ?? string.Empty)
                .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
            int selectedIndex = Array.FindIndex(entries, entry => string.Equals(entry, selectedCudaBin, StringComparison.OrdinalIgnoreCase));
            int fallbackIndex = Array.FindIndex(entries, entry => string.Equals(entry, fallbackCudnnBin, StringComparison.OrdinalIgnoreCase));

            Assert.True(selectedIndex >= 0);
            Assert.True(fallbackIndex >= 0);
            Assert.True(selectedIndex < fallbackIndex, $"Selected CUDA directory index {selectedIndex} must precede fallback index {fallbackIndex}.");
            Assert.Equal(1, entries.Count(entry => string.Equals(entry, selectedCudaBin, StringComparison.OrdinalIgnoreCase)));
        }
        finally
        {
            Environment.SetEnvironmentVariable("PATH", previousPath);
            Environment.SetEnvironmentVariable("JYPPX_CUDA_ROOT", previousCudaRoot);
            Environment.SetEnvironmentVariable("JYPPX_CUDNN_ROOT", previousCudnnRoot);
            Environment.SetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING", previousDevelopmentProbing);
        }
    }

    [Fact]
    public void ExplicitBridgeDirectoryIsUsedAsDependencyDirectory()
    {
        string previous = Environment.GetEnvironmentVariable("JYPPX_NATIVE_BRIDGE_PATH") ?? string.Empty;
        string bridgeDirectory = Path.Combine(RepositoryPaths.Root, "build-out", "win-x64-trt10-cuda11-release", "bin", "Release");

        try
        {
            Directory.CreateDirectory(bridgeDirectory);
            Environment.SetEnvironmentVariable("JYPPX_NATIVE_BRIDGE_PATH", bridgeDirectory);
            string[] directories = NativeBridgePathResolver.EnumerateDependencyDirectories(Assembly.GetExecutingAssembly()).ToArray();

            Assert.Contains(bridgeDirectory, directories);
        }
        finally
        {
            Environment.SetEnvironmentVariable("JYPPX_NATIVE_BRIDGE_PATH", string.IsNullOrEmpty(previous) ? null : previous);
        }
    }

    [Fact]
    public void CurrentDirectoryIsNotAProductionCandidatePath()
    {
        string previous = Environment.GetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING") ?? string.Empty;
        string previousDirectory = Directory.GetCurrentDirectory();
        string probeDirectory = Path.Combine(RepositoryPaths.Root, "build-out", "path-resolver-test", Guid.NewGuid().ToString("N"));

        try
        {
            Directory.CreateDirectory(probeDirectory);
            Directory.SetCurrentDirectory(probeDirectory);
            Environment.SetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING", null);
            string currentDirectoryCandidate = Path.Combine(probeDirectory, NativeBridgePathResolver.GetBridgeFileName());
            string[] candidates = NativeBridgePathResolver.EnumerateCandidatePaths(Assembly.GetExecutingAssembly()).ToArray();

            Assert.DoesNotContain(currentDirectoryCandidate, candidates);
        }
        finally
        {
            Directory.SetCurrentDirectory(previousDirectory);
            Environment.SetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING", string.IsNullOrEmpty(previous) ? null : previous);
        }
    }

    [Fact]
    public void DevelopmentProbingIncludesCurrentDirectoryCandidatePath()
    {
        string previous = Environment.GetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING") ?? string.Empty;
        string previousDirectory = Directory.GetCurrentDirectory();
        string probeDirectory = Path.Combine(RepositoryPaths.Root, "build-out", "path-resolver-test", Guid.NewGuid().ToString("N"));

        try
        {
            Directory.CreateDirectory(probeDirectory);
            Directory.SetCurrentDirectory(probeDirectory);
            Environment.SetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING", "1");
            string currentDirectoryCandidate = Path.Combine(probeDirectory, NativeBridgePathResolver.GetBridgeFileName());
            string[] candidates = NativeBridgePathResolver.EnumerateCandidatePaths(Assembly.GetExecutingAssembly()).ToArray();

            Assert.Contains(currentDirectoryCandidate, candidates);
        }
        finally
        {
            Directory.SetCurrentDirectory(previousDirectory);
            Environment.SetEnvironmentVariable("JYPPX_ENABLE_DEVELOPMENT_PROBING", string.IsNullOrEmpty(previous) ? null : previous);
        }
    }
}
