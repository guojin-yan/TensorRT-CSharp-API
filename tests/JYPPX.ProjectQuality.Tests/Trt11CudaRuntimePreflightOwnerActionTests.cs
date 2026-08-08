using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt11CudaRuntimePreflightOwnerActionTests
{
    [Fact]
    public void RuntimeConsumerScriptDefinesCudaPreflightMarkersAndProofBoundary()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-BridgePackageRuntimeConsumer.ps1"));

        foreach (string marker in new[]
        {
            "WriteCudaPreflight",
            "CudaPreflightAvailable=",
            "CudaPreflightAttempted=",
            "CudaPreflightDriverVersion=",
            "CudaPreflightRuntimeVersion=",
            "CudaPreflightDeviceCount=",
            "CudaPreflightGetDeviceCountStatus=",
            "CudaPreflightInitStatus=",
            "CudaPreflightLastErrorName=",
            "CudaPreflightLastErrorMessage=",
            "CudaPreflightCanAttemptTensorRtRuntimeCreate=",
            "Join-Path $Roots.CudaRoot \"bin\\x64\"",
            "cudaPreflight = [ordered]@",
            "cannot promote runtime proof by itself"
        })
        {
            Assert.Contains(marker, script, StringComparison.Ordinal);
        }
    }
}
