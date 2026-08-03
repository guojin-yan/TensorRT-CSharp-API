using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeManifestTests
{
    private static readonly string[] SplitRuntimeRoles = ["bridge", "cuda-cudnn", "tensorrt"];

    [Fact]
    public void PublicRuntimeManifestDoesNotContainLocalWindowsRoots()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        foreach (JsonElement package in document.RootElement.GetProperty("packages").EnumerateArray())
        {
            if (package.GetProperty("platform").GetString() != "windows")
            {
                continue;
            }

            Assert.False(HasLocalAbsolutePath(package, "defaultTensorRtRoot"), $"{package.GetProperty("key").GetString()} must use runtime-packages.local.json for TensorRT roots.");
            Assert.False(HasLocalAbsolutePath(package, "defaultCudaRoot"), $"{package.GetProperty("key").GetString()} must use runtime-packages.local.json for CUDA roots.");
        }
    }

    [Fact]
    public void RetiredFullRuntimeProjectsAreAbsent()
    {
        string root = Path.Combine(RepositoryPaths.Root, "pack", "runtime");
        Assert.Empty(Directory.GetFiles(root, "*.csproj", SearchOption.AllDirectories));
    }

    [Fact]
    public void LinuxRuntimePackagesDeclareDistributionAndArchitecture()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        JsonElement[] linuxPackages = document.RootElement.GetProperty("packages")
            .EnumerateArray()
            .Where(static package => package.GetProperty("platform").GetString() == "linux")
            .ToArray();

        Assert.True(linuxPackages.Length >= 6, "Linux runtime packages must cover more than the single TensorRT 11 CUDA 12.9 package.");

        foreach (JsonElement package in linuxPackages)
        {
            string key = package.GetProperty("key").GetString()!;
            string packageId = package.GetProperty("packageId").GetString()!;
            string distro = package.GetProperty("linuxDistro").GetString()!;
            string distroVersion = package.GetProperty("linuxDistroVersion").GetString()!;
            string architecture = package.GetProperty("architecture").GetString()!;
            string runnerMode = package.GetProperty("runnerMode").GetString()!;

            Assert.Equal("ubuntu", distro);
            Assert.Contains(architecture, new[] { "x64", "arm64" });
            Assert.Contains(runnerMode, new[] { "hosted", "hosted-container", "self-hosted" });
            Assert.Contains($"linux-{architecture}-ubuntu{distroVersion}-", key);
            Assert.Contains($".linux-{architecture}.ubuntu{distroVersion}.", packageId);
        }
    }

    [Fact]
    public void Ubuntu2204LinuxRuntimeMatrixCoversAllConfiguredDependencyCombinations()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        string[] expectedSuffixes =
        [
            "trt8.6-cuda11.8-cudnn8.9",
            "trt8.6-cuda12.1-cudnn8.9",
            "trt10.11-cuda11.8-cudnn8.9",
            "trt10.11-cuda12.9-cudnn9.22",
            "trt11.0-cuda12.9-cudnn9.22",
            "trt11.0-cuda13.2-cudnn9.22",
        ];

        HashSet<string> ubuntu2204Keys = document.RootElement.GetProperty("packages")
            .EnumerateArray()
            .Where(static package =>
                package.GetProperty("platform").GetString() == "linux" &&
                package.GetProperty("linuxDistroVersion").GetString() == "22.04" &&
                package.GetProperty("architecture").GetString() == "x64")
            .Select(static package => package.GetProperty("key").GetString()!)
            .ToHashSet(StringComparer.Ordinal);

        foreach (string suffix in expectedSuffixes)
        {
            Assert.Contains("linux-x64-ubuntu22.04-" + suffix, ubuntu2204Keys);
        }
    }

    [Fact]
    public void LinuxRuntimeTargetCatalogSeparatesModeledAndFuturePackageLines()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "linux-runtime-targets.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        string[] dependencyCombinations = document.RootElement.GetProperty("dependencyCombinations")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Equal(6, dependencyCombinations.Length);
        Assert.Contains("trt8.6-cuda11.8-cudnn8.9", dependencyCombinations);
        Assert.Contains("trt11.0-cuda13.2-cudnn9.22", dependencyCombinations);

        JsonElement[] modeledTargets = document.RootElement.GetProperty("targets").EnumerateArray().ToArray();
        Assert.Equal(3, modeledTargets.Length);
        Assert.Contains(modeledTargets, static target => target.GetProperty("target").GetString() == "ubuntu22.04-x64-hosted");
        Assert.Contains(modeledTargets, static target => target.GetProperty("target").GetString() == "ubuntu24.04-x64-hosted");
        Assert.Contains(modeledTargets, static target => target.GetProperty("target").GetString() == "ubuntu20.04-x64-hosted-container");

        foreach (JsonElement target in modeledTargets)
        {
            Assert.Equal("modeled", target.GetProperty("status").GetString());
            Assert.True(target.GetProperty("expectedCombinations").GetArrayLength() > 0);
            Assert.True(target.GetProperty("keySetAliases").GetArrayLength() > 0);
        }

        JsonElement[] futureTargets = document.RootElement.GetProperty("futureTargets").EnumerateArray().ToArray();
        Assert.Equal(3, futureTargets.Length);
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "linux-arm64-sbsa");
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "linux-jetson-l4t");
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "non-ubuntu-linux");

        foreach (JsonElement target in futureTargets)
        {
            Assert.Equal("future-separate-package-line", target.GetProperty("status").GetString());
            Assert.False(string.IsNullOrWhiteSpace(target.GetProperty("packageIdentityRule").GetString()));
            Assert.True(target.GetProperty("requiredEvidenceItems").GetArrayLength() > 0);
            Assert.True(target.GetProperty("keySetAliases").GetArrayLength() > 0);
        }
    }

    [Fact]
    public void SplitRuntimeProjectsExistOnlyForBridgePackages()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime-split", "split-runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        foreach (JsonElement package in document.RootElement.GetProperty("packages").EnumerateArray())
        {
            string key = package.GetProperty("key").GetString()!;
            string packageId = package.GetProperty("packageId").GetString()!;
            string role = package.GetProperty("role").GetString()!;
            string prototypeState = package.GetProperty("prototypeState").GetString()!;
            string projectPath = Path.Combine(RepositoryPaths.Root, "pack", "runtime-split", key, packageId + ".csproj");

            if (role == "bridge")
            {
                Assert.True(File.Exists(projectPath), $"Bridge runtime project is missing for {key}: {projectPath}");
            }
            else
            {
                Assert.False(File.Exists(projectPath), $"Retired vendor runtime project must be removed: {projectPath}");
            }

            Assert.Contains(role, SplitRuntimeRoles);
            Assert.Contains(prototypeState, new[] { "design-only", "local-validated", "pending-local-validation" });
        }

        string splitRoot = Path.Combine(RepositoryPaths.Root, "pack", "runtime-split");
        Assert.All(
            Directory.GetFiles(splitRoot, "*.csproj", SearchOption.AllDirectories),
            static projectPath => Assert.EndsWith(".Bridge.csproj", projectPath, StringComparison.Ordinal));
    }

    [Fact]
    public void SplitRuntimeAssetsCoverSourceRuntimeAssetsWithoutOverlap()
    {
        string runtimeManifestPath = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-packages.manifest.json");
        string splitManifestPath = Path.Combine(RepositoryPaths.Root, "pack", "runtime-split", "split-runtime-packages.manifest.json");
        using JsonDocument runtimeManifest = JsonDocument.Parse(File.ReadAllText(runtimeManifestPath));
        using JsonDocument splitManifest = JsonDocument.Parse(File.ReadAllText(splitManifestPath));

        Dictionary<string, JsonElement> sourcePackages = runtimeManifest.RootElement.GetProperty("packages")
            .EnumerateArray()
            .ToDictionary(package => package.GetProperty("key").GetString()!, package => package.Clone());

        var groupedSplitPackages = splitManifest.RootElement.GetProperty("packages")
            .EnumerateArray()
            .GroupBy(package => package.GetProperty("sourceRuntimeKey").GetString()!)
            .ToArray();

        foreach (IGrouping<string, JsonElement> group in groupedSplitPackages)
        {
            Assert.True(sourcePackages.TryGetValue(group.Key, out JsonElement sourcePackage), $"Split source package is missing from runtime manifest: {group.Key}");

            string[] fullAssets = GetExpectedRuntimeAssetNames(sourcePackage).OrderBy(static value => value, StringComparer.Ordinal).ToArray();
            string[] splitAssets = group.SelectMany(GetSplitAssets).OrderBy(static value => value, StringComparer.Ordinal).ToArray();
            string[] distinctSplitAssets = splitAssets.Distinct(StringComparer.Ordinal).OrderBy(static value => value, StringComparer.Ordinal).ToArray();

            foreach (string fullAsset in fullAssets)
            {
                Assert.Contains(distinctSplitAssets, splitAsset => AssetPatternComparer.Instance.Equals(fullAsset, splitAsset));
            }

            foreach (string splitAsset in distinctSplitAssets)
            {
                Assert.Contains(fullAssets, fullAsset => AssetPatternComparer.Instance.Equals(fullAsset, splitAsset));
            }

            Assert.Equal(splitAssets.Length, splitAssets.Distinct(StringComparer.Ordinal).Count());
        }
    }

    private static bool HasLocalAbsolutePath(JsonElement package, string propertyName)
    {
        if (!package.TryGetProperty(propertyName, out JsonElement value))
        {
            return false;
        }

        string? text = value.GetString();
        return !string.IsNullOrWhiteSpace(text) && text.Length >= 3 && char.IsLetter(text[0]) && text[1] == ':' && (text[2] == '\\' || text[2] == '/');
    }

    private static IEnumerable<string> GetSplitAssets(JsonElement package)
    {
        foreach (JsonElement asset in package.GetProperty("assets").EnumerateArray())
        {
            yield return asset.GetString()!;
        }
    }

    private static IEnumerable<string> GetExpectedRuntimeAssetNames(JsonElement package)
    {
        yield return package.GetProperty("bridgeFile").GetString()!;

        foreach (JsonElement asset in package.GetProperty("tensorRtFiles").EnumerateArray())
        {
            yield return Path.GetFileName(asset.GetString()!);
        }

        foreach (JsonElement asset in package.GetProperty("cudaFiles").EnumerateArray())
        {
            yield return Path.GetFileName(asset.GetString()!);
        }

        foreach (JsonElement asset in package.GetProperty("cudnnFiles").EnumerateArray())
        {
            yield return Path.GetFileName(asset.GetString()!);
        }
    }

    private sealed class AssetPatternComparer : IEqualityComparer<string>
    {
        public static AssetPatternComparer Instance { get; } = new();

        public bool Equals(string? x, string? y)
        {
            if (string.Equals(x, y, StringComparison.Ordinal))
            {
                return true;
            }

            if (x == null || y == null)
            {
                return false;
            }

            if (x.Contains('*', StringComparison.Ordinal))
            {
                return MatchesPattern(x, y);
            }

            if (y.Contains('*', StringComparison.Ordinal))
            {
                return MatchesPattern(y, x);
            }

            return false;
        }

        public int GetHashCode(string obj)
        {
            return 0;
        }

        private static bool MatchesPattern(string pattern, string value)
        {
            string regex = "^" + System.Text.RegularExpressions.Regex.Escape(pattern).Replace("\\*", ".*") + "$";
            return System.Text.RegularExpressions.Regex.IsMatch(value, regex, System.Text.RegularExpressions.RegexOptions.CultureInvariant);
        }
    }
}
