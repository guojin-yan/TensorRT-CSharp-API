using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt11CompatibleHostRuntimeProofTests
{
    [Fact]
    public void PluginInventoryFieldPolicyPreservesDefaultsAndFlowsThroughPublicCopiedSnapshots()
    {
        string environment = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs");
        string globalInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.GlobalRuntimePluginProbe.cs");
        string capabilityInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs");

        Assert.Contains("GetGlobalPluginRegistryInventory(line, includeCreatorFields: true)", environment, StringComparison.Ordinal);
        Assert.Contains("TryGetGlobalPluginRegistryInventory(line, includeCreatorFields: true", environment, StringComparison.Ordinal);
        Assert.Contains("includeCreatorFields: true,\n            out creator", NormalizeNewlines(environment), StringComparison.Ordinal);
        Assert.Contains("GetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields: true)", environment, StringComparison.Ordinal);
        Assert.Contains("if (includeCreatorFields)", globalInterop, StringComparison.Ordinal);
        Assert.Contains("if (includeCreatorFields)", capabilityInterop, StringComparison.Ordinal);
        Assert.Contains("Array.Empty<TensorRtPluginFieldInfo>()", globalInterop + capabilityInterop, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", environment + globalInterop + capabilityInterop, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", environment + globalInterop + capabilityInterop, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", environment + globalInterop + capabilityInterop, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRt11SmokesAvoidParserOnlyFieldHooksAndDowngradeRemovedCapabilityQueries()
    {
        string pluginSmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string onnxSmoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("bool includeCreatorFields = line != TensorRtApiLine.TensorRt11;", pluginSmoke, StringComparison.Ordinal);
        Assert.Contains("CreatorFieldCollection Included={includeCreatorFields}", pluginSmoke, StringComparison.Ordinal);
        Assert.Contains("includeCreatorFields: line != TensorRtApiLine.TensorRt11", pluginSmoke, StringComparison.Ordinal);
        Assert.Contains("ProbeBuilderCapabilities(builder)", onnxSmoke, StringComparison.Ordinal);
        Assert.Contains("catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.NotSupported)", onnxSmoke, StringComparison.Ordinal);
        Assert.Contains("Unavailable:{exception.StatusCode}", onnxSmoke, StringComparison.Ordinal);
    }

    [Fact]
    public void BridgeOnlyConsumerCompilesEveryFieldPolicyOverloadWithoutProjectReferences()
    {
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("GetGlobalPluginRegistryInventory(line, includeCreatorFields)", consumer, StringComparison.Ordinal);
        Assert.Contains("TryGetGlobalPluginRegistryInventory(line, includeCreatorFields", consumer, StringComparison.Ordinal);
        Assert.Contains("TryGetGlobalPluginCreator(line, name, version, pluginNamespace, includeCreatorFields", consumer, StringComparison.Ordinal);
        Assert.Contains("GetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields)", consumer, StringComparison.Ordinal);
        Assert.Contains("TryGetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields", consumer, StringComparison.Ordinal);
        Assert.Contains("TryGetBuilderCapabilityPluginCreator(line, capability, name, version, pluginNamespace, includeCreatorFields", consumer, StringComparison.Ordinal);
        Assert.DoesNotContain("<ProjectReference Include=", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void CompatibleHostProofExporterPinsHashesAndForbidsReleaseSideEffects()
    {
        string exporter = ReadSource("eng", "Export-Trt11Cuda129CompatibleHostProof.ps1");

        Assert.Contains("F1E1E896B5066472DD900CBD830950781E2215D967BA47BC97B3D103377FD0F3", exporter, StringComparison.Ordinal);
        Assert.Contains("93E8CA4FD0B95CFB49C3E2CDC6BB94AFA8126853BE66D0B5013D60875C325A2C", exporter, StringComparison.Ordinal);
        Assert.Contains("FE26D320160AF0B8CCD76F044429CE106DF8D89FE89B62859693FBC80FCD79CB", exporter, StringComparison.Ordinal);
        Assert.Contains("Unexpected Internal Error", exporter, StringComparison.Ordinal);
        Assert.Contains("compatible-host-bridge-package-runtime", exporter, StringComparison.Ordinal);
        Assert.Contains("isPackageConsumerRuntimeProof = $false", exporter, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", exporter, StringComparison.Ordinal);
        Assert.Contains("pushesNuGet = $false", exporter, StringComparison.Ordinal);
        Assert.Contains("uploadsGitHubRelease = $false", exporter, StringComparison.Ordinal);
        Assert.Contains("closesIssue = $false", exporter, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", exporter, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", exporter, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static string NormalizeNewlines(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal);
    }
}
