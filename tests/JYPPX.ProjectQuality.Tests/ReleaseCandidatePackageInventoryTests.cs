using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidatePackageInventoryTests
{
    [Fact]
    public void InventoryPolicyAllowsManagedAndBridgeOnlyPackagesWithoutPublishing()
    {
        string script = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Export-ReleaseCandidatePackageInventory.ps1"));

        foreach (string marker in new[]
        {
            "recordKind = \"release-candidate-package-inventory\"",
            "release-candidate-package-inventory.json",
            "Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256",
            "role = $role",
            "managedPackageReady",
            "splitBridgePackageReady",
            "requiredSplitRoles = @(\"split-bridge\")",
            "fullRuntimePackageRequired = $false",
            "vendorRuntimePackagesForbidden = $true",
            "canPublishPublicly = $false",
            "canUseAsPublicPackageProof = $false",
            "Local package inventory accepts only the core managed API and bridge candidates"
        })
        {
            Assert.Contains(marker, script, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release create", script, StringComparison.OrdinalIgnoreCase);
    }
}
