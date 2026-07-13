using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaDeviceInitializationProofRunnerTests
{
    [Fact]
    public void ProofRunnerExistsAndKeepsPreInitCallOrder()
    {
        string project = ReadSource("smoke", "CudaDeviceInitializationProofRunner", "CudaDeviceInitializationProofRunner.csproj");
        string program = ReadSource("smoke", "CudaDeviceInitializationProofRunner", "Program.cs");

        Assert.Contains("<ProjectReference Include=\"..\\..\\src\\JYPPX.CudaSharp\\JYPPX.CudaSharp.csproj\" />", project);
        Assert.Contains("CudaDeviceInitializationProof Start Device=", program);
        Assert.Contains("ProofKind=local-smoke-not-external-proof", program);
        Assert.Contains("IsPackageConsumerRuntimeProof=False", program);
        Assert.Contains("CanPromoteRuntimeProof=False", program);

        int setValidDevicesIndex = program.IndexOf("CudaDevice.SetValidDevices(new[] { deviceOrdinal })", StringComparison.Ordinal);
        int initDeviceIndex = program.IndexOf("CudaDevice.InitDevice(deviceOrdinal, CudaDeviceRuntimeFlags.ScheduleAuto)", StringComparison.Ordinal);
        int chooseDeviceIndex = program.IndexOf("CudaDevice.ChooseDevice(requirements)", StringComparison.Ordinal);

        Assert.True(setValidDevicesIndex >= 0, "Proof runner must call SetValidDevices before CUDA context creation.");
        Assert.True(initDeviceIndex > setValidDevicesIndex, "Proof runner must call InitDevice after SetValidDevices.");
        Assert.True(chooseDeviceIndex > initDeviceIndex, "Proof runner must call ChooseDevice after InitDevice.");

        Assert.DoesNotContain("CudaDevice.Count", program);
        Assert.DoesNotContain("CudaDevice.Current", program);
        Assert.DoesNotContain("CudaEnvironmentProbe.GetCurrent", program);
    }

    [Fact]
    public void ProofRunnerReportsSuccessAndSkipWithoutFakingExternalProof()
    {
        string program = ReadSource("smoke", "CudaDeviceInitializationProofRunner", "Program.cs");

        Assert.Contains("SetValidDevices=Ok Count=1", program);
        Assert.Contains("InitDevice=Ok Device=", program);
        Assert.Contains("ChooseDevice=Ok Device=", program);
        Assert.Contains("Skipped=True Reason=CudaException:", program);
        Assert.Contains("Skipped=True Reason=DllNotFoundException:", program);
        Assert.Contains("Skipped=True Reason=BadImageFormatException:", program);
        Assert.Contains("CudaDeviceInitializationProof Completed=True", program);
        Assert.Contains("CudaDeviceInitializationProof Completed=False", program);

        Assert.DoesNotContain("IsPackageConsumerRuntimeProof=True", program);
        Assert.DoesNotContain("CanPromoteRuntimeProof=True", program);
        Assert.DoesNotContain("post-publish-proof", program, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
