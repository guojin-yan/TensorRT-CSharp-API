using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeManifestTests
{
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
    public void RuntimeProjectsExistForEveryManifestPackage()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        foreach (JsonElement package in document.RootElement.GetProperty("packages").EnumerateArray())
        {
            string key = package.GetProperty("key").GetString()!;
            string packageId = package.GetProperty("packageId").GetString()!;
            string projectPath = Path.Combine(RepositoryPaths.Root, "pack", "runtime", key, packageId + ".csproj");

            Assert.True(File.Exists(projectPath), $"Runtime project is missing for {key}: {projectPath}");
        }
    }

    [Fact]
    public void SplitRuntimeProjectsExistForEverySplitManifestPackage()
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

            Assert.True(File.Exists(projectPath), $"Split runtime project is missing for {key}: {projectPath}");
            Assert.False(string.IsNullOrWhiteSpace(role), $"Split runtime role is invalid for {key}: {role}");
            Assert.Contains(prototypeState, new[] { "design-only", "local-validated" });
        }
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
            .Where(package =>
            {
                string? tier = package.GetProperty("distributionTier").GetString();
                return tier is "split-delivery-candidate" or "private-feed";
            })
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
