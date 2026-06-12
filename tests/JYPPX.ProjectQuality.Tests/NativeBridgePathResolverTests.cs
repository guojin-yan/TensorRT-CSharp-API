using System.Reflection;
using JYPPX.Shared.Interop;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class NativeBridgePathResolverTests
{
    [Fact]
    public void ExplicitBridgeDirectoryIsUsedAsDependencyDirectory()
    {
        string previous = Environment.GetEnvironmentVariable("JYPPX_NATIVE_BRIDGE_PATH") ?? string.Empty;
        string bridgeDirectory = Path.Combine(RepositoryPaths.Root, "build-out", "win-x64-trt10-cuda11-release", "bin", "Release");

        try
        {
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
