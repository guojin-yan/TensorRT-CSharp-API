using JYPPX.CudaSharp;
using System.Security.Cryptography;

namespace CudaRuntimeCompilationSample;

internal static class Program
{
    public static int Main()
    {
        const string sourceText = """
#include "scale.cuh"

extern "C" __global__ void vector_add(const float* left, const float* right, float* output, int count)
{
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
    {
        output[index] = (left[index] + right[index]) * SAMPLE_SCALE;
    }
}

template <typename T>
__global__ void typed_identity(T* values)
{
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    values[index] = values[index];
}
""";

        var source = new CudaRtcProgramSource(
            sourceText,
            "runtime-compilation-sample.cu",
            new[] { new CudaRtcHeader("scale.cuh", "#define SAMPLE_SCALE 2.0f\n") },
            new[] { "&typed_identity<float>" });
        var options = new CudaRtcCompileOptions(
            targetArchitecture: "compute_75",
            generateLineInfo: true);

        CudaRtcCapability capability = CudaRtcCompiler.GetCapability();
        Console.WriteLine($"capability.available={capability.IsAvailable}");
        Console.WriteLine($"capability.version={capability.Version}");
        Console.WriteLine($"capability.library={capability.LoadedLibraryName}");
        Console.WriteLine($"capability.ptx={capability.SupportsPtx} cubin={capability.SupportsCubin} ltoIr={capability.SupportsLtoIr} nvvm={capability.SupportsDeprecatedNvvm}");
        if (!capability.IsAvailable)
        {
            Console.Error.WriteLine(capability.DependencyDiagnostic);
            return 2;
        }

        CudaDriverCapability driverCapability = CudaDriver.GetCapability();
        Console.WriteLine($"driver.capability.available={driverCapability.IsAvailable} version={driverCapability.DriverVersion} library={driverCapability.LoadedLibraryName} moduleLoad={driverCapability.SupportsModuleLoad} typedLaunch={driverCapability.SupportsTypedLaunch} contextInterop={driverCapability.SupportsContextInterop} completionEvents={driverCapability.SupportsCompletionEvents}");
        if (!driverCapability.IsAvailable)
        {
            Console.Error.WriteLine(driverCapability.DependencyDiagnostic);
            return 10;
        }

        CudaRtcCompilationResult success = CudaRtcCompiler.Compile(source, options);
        if (!success.Success)
        {
            Console.Error.WriteLine(success.Log);
            return 3;
        }

        CudaRtcArtifact? ptx = success.FindArtifact(CudaRtcArtifactKind.Ptx);
        if (ptx == null || success.LoweredNames.Count != 1 || string.IsNullOrEmpty(success.LoweredNames[0].LoweredName))
        {
            Console.Error.WriteLine("Successful compilation did not produce the required PTX/lowered-name copies.");
            return 4;
        }

        CudaRtcCompilationResult repeated = CudaRtcCompiler.Compile(source, options);
        CudaRtcArtifact? repeatedPtx = repeated.FindArtifact(CudaRtcArtifactKind.Ptx);
        if (repeatedPtx == null || !string.Equals(ptx.Sha256, repeatedPtx.Sha256, StringComparison.Ordinal))
        {
            Console.Error.WriteLine("Repeated NVRTC compilation did not preserve the PTX SHA256 determinism contract.");
            return 6;
        }
        Console.WriteLine($"determinism.ptxSha256={ptx.Sha256} repeated=True");

        CudaRtcCompilationResult cubinResult = CudaRtcCompiler.Compile(
            source,
            new CudaRtcCompileOptions(targetArchitecture: "sm_75"));
        CudaRtcArtifact? cubin = cubinResult.FindArtifact(CudaRtcArtifactKind.Cubin);
        if (!cubinResult.Success || cubin == null)
        {
            Console.Error.WriteLine("A real SM target did not produce the required copied CUBIN artifact.");
            return 7;
        }
        Console.WriteLine($"artifact.kind={cubin.Kind} bytes={cubin.Length} sha256={cubin.Sha256} target={cubin.TargetArchitecture}");

        if (capability.SupportsLtoIr)
        {
            CudaRtcCompilationResult ltoResult = CudaRtcCompiler.Compile(
                source,
                new CudaRtcCompileOptions(targetArchitecture: "compute_75", emitLtoIr: true));
            CudaRtcArtifact? lto = ltoResult.FindArtifact(CudaRtcArtifactKind.LtoIr);
            if (!ltoResult.Success || lto == null)
            {
                Console.Error.WriteLine("An LTO-enabled compilation did not produce the required copied LTO IR artifact.");
                return 8;
            }
            Console.WriteLine($"artifact.kind={lto.Kind} bytes={lto.Length} sha256={lto.Sha256} target={lto.TargetArchitecture}");
        }

        Console.WriteLine($"compile.success=True result={success.ResultCode} logLength={success.Log.Length}");
        Console.WriteLine($"lowered.expression={success.LoweredNames[0].Expression}");
        Console.WriteLine($"lowered.name={success.LoweredNames[0].LoweredName}");
        foreach (CudaRtcArtifact artifact in success.Artifacts)
        {
            Console.WriteLine($"artifact.kind={artifact.Kind} bytes={artifact.Length} sha256={artifact.Sha256} sourceSha256={artifact.SourceSha256} optionsSha256={artifact.OptionsSha256}");
        }

        bool loadSucceeded = false;
        string loadDiagnostic = string.Empty;
        bool launchAttempted = false;
        bool launchSucceeded = false;
        bool gpuReadback = false;
        bool correctnessProof = false;
        bool completedBeforeSynchronize = false;
        bool ownersDisposedBeforeSynchronize = false;
        float maxAbsoluteError = 0;
        string outputSha256 = string.Empty;
        string launchDiagnostic = string.Empty;
        try
        {
            using CudaKernelLibrary library = CudaKernelLibrary.Load(ptx.ToArray());
            loadSucceeded = library.ContainsKernel("vector_add");
            if (!loadSucceeded)
            {
                loadDiagnostic = "PTX loaded, but vector_add was not found in the copied library inventory query.";
            }
            else
            {
                const int elementCount = 257;
                const int threadsPerBlock = 128;
                float[] leftValues = Enumerable.Range(0, elementCount).Select(index => index * 0.25f).ToArray();
                float[] rightValues = Enumerable.Range(0, elementCount).Select(index => 100.0f - index * 0.125f).ToArray();
                using var left = new CudaMemory(elementCount * sizeof(float));
                using var right = new CudaMemory(elementCount * sizeof(float));
                using var output = new CudaMemory(elementCount * sizeof(float));
                using var stream = new CudaStream();
                left.CopyFrom(leftValues);
                right.CopyFrom(rightValues);
                output.Fill(0);

                launchAttempted = true;
                var launchConfiguration = new CudaKernelLaunchConfiguration(
                    new CudaDim3((uint)((elementCount + threadsPerBlock - 1) / threadsPerBlock)),
                    new CudaDim3(threadsPerBlock));
                using CudaKernelLaunch launch = library.Launch(
                    "vector_add",
                    launchConfiguration,
                    stream,
                    CudaKernelArgument.FromDeviceMemory(left),
                    CudaKernelArgument.FromDeviceMemory(right),
                    CudaKernelArgument.FromDeviceMemory(output),
                    CudaKernelArgument.FromInt32(elementCount));
                completedBeforeSynchronize = launch.IsCompleted;
                library.Dispose();
                stream.Dispose();
                left.Dispose();
                right.Dispose();
                ownersDisposedBeforeSynchronize = true;
                launch.Synchronize();
                launchSucceeded = launch.IsCompleted;

                float[] actual = output.ToSingleArray(elementCount);
                gpuReadback = actual.Length == elementCount;
                for (int index = 0; index < actual.Length; ++index)
                {
                    float expected = (leftValues[index] + rightValues[index]) * 2.0f;
                    maxAbsoluteError = Math.Max(maxAbsoluteError, Math.Abs(actual[index] - expected));
                }
                correctnessProof = gpuReadback && maxAbsoluteError <= 1e-6f;
                byte[] outputBytes = new byte[actual.Length * sizeof(float)];
                Buffer.BlockCopy(actual, 0, outputBytes, 0, outputBytes.Length);
                outputSha256 = Convert.ToHexString(SHA256.HashData(outputBytes)).ToLowerInvariant();
            }
        }
        catch (CudaException exception)
        {
            if (loadSucceeded)
            {
                launchDiagnostic = exception.Message;
            }
            else
            {
                loadDiagnostic = exception.Message;
            }
        }
        Console.WriteLine($"load.attempted=True load.succeeded={loadSucceeded} classification=local-toolkit-compile-to-load");
        if (loadDiagnostic.Length != 0)
        {
            Console.WriteLine("load.diagnostic=" + loadDiagnostic.Replace(Environment.NewLine, " | "));
        }
        Console.WriteLine($"launch.attempted={launchAttempted} launch.succeeded={launchSucceeded} completedBeforeSynchronize={completedBeforeSynchronize} ownersDisposedBeforeSynchronize={ownersDisposedBeforeSynchronize} gpuReadback={gpuReadback} correctness={correctnessProof} maxAbsoluteError={maxAbsoluteError:G9} outputSha256={outputSha256}");
        if (launchDiagnostic.Length != 0)
        {
            Console.WriteLine("launch.diagnostic=" + launchDiagnostic.Replace(Environment.NewLine, " | "));
        }
        if (loadSucceeded && (!launchSucceeded || !gpuReadback || !correctnessProof))
        {
            Console.Error.WriteLine("A loadable RTC artifact did not complete owner-bound launch/readback correctness validation.");
            return 9;
        }

        bool driverLoadSucceeded = false;
        bool driverLaunchAttempted = false;
        bool driverLaunchSucceeded = false;
        bool driverGpuReadback = false;
        bool driverCorrectnessProof = false;
        bool driverCompletedBeforeSynchronize = false;
        bool driverOwnersDisposedBeforeSynchronize = false;
        float driverMaxAbsoluteError = 0;
        string driverOutputSha256 = string.Empty;
        string driverLoadDiagnostic = string.Empty;
        string driverLaunchDiagnostic = string.Empty;
        try
        {
            using CudaDriverModule module = CudaDriverModule.Load(ptx.ToArray());
            driverLoadSucceeded = true;
            const int elementCount = 257;
            const int threadsPerBlock = 128;
            float[] leftValues = Enumerable.Range(0, elementCount).Select(index => index * 0.25f).ToArray();
            float[] rightValues = Enumerable.Range(0, elementCount).Select(index => 100.0f - index * 0.125f).ToArray();
            using var left = new CudaMemory(elementCount * sizeof(float));
            using var right = new CudaMemory(elementCount * sizeof(float));
            using var output = new CudaMemory(elementCount * sizeof(float));
            using var stream = new CudaStream();
            left.CopyFrom(leftValues);
            right.CopyFrom(rightValues);
            output.Fill(0);

            driverLaunchAttempted = true;
            var configuration = new CudaKernelLaunchConfiguration(
                new CudaDim3((uint)((elementCount + threadsPerBlock - 1) / threadsPerBlock)),
                new CudaDim3(threadsPerBlock));
            using CudaDriverKernelLaunch launch = module.Launch(
                "vector_add",
                configuration,
                stream,
                CudaKernelArgument.FromDeviceMemory(left),
                CudaKernelArgument.FromDeviceMemory(right),
                CudaKernelArgument.FromDeviceMemory(output),
                CudaKernelArgument.FromInt32(elementCount));
            driverCompletedBeforeSynchronize = launch.IsCompleted;
            module.Dispose();
            stream.Dispose();
            left.Dispose();
            right.Dispose();
            driverOwnersDisposedBeforeSynchronize = true;
            launch.Synchronize();
            driverLaunchSucceeded = launch.IsCompleted;

            float[] actual = output.ToSingleArray(elementCount);
            driverGpuReadback = actual.Length == elementCount;
            for (int index = 0; index < actual.Length; ++index)
            {
                float expected = (leftValues[index] + rightValues[index]) * 2.0f;
                driverMaxAbsoluteError = Math.Max(driverMaxAbsoluteError, Math.Abs(actual[index] - expected));
            }
            driverCorrectnessProof = driverGpuReadback && driverMaxAbsoluteError <= 1e-6f;
            byte[] outputBytes = new byte[actual.Length * sizeof(float)];
            Buffer.BlockCopy(actual, 0, outputBytes, 0, outputBytes.Length);
            driverOutputSha256 = Convert.ToHexString(SHA256.HashData(outputBytes)).ToLowerInvariant();
        }
        catch (CudaException exception)
        {
            if (driverLoadSucceeded)
            {
                driverLaunchDiagnostic = exception.Message;
            }
            else
            {
                driverLoadDiagnostic = exception.Message;
            }
        }
        Console.WriteLine($"driver.load.attempted=True driver.load.succeeded={driverLoadSucceeded}");
        if (driverLoadDiagnostic.Length != 0)
        {
            Console.WriteLine("driver.load.diagnostic=" + driverLoadDiagnostic.Replace(Environment.NewLine, " | "));
        }
        Console.WriteLine($"driver.launch.attempted={driverLaunchAttempted} driver.launch.succeeded={driverLaunchSucceeded} completedBeforeSynchronize={driverCompletedBeforeSynchronize} ownersDisposedBeforeSynchronize={driverOwnersDisposedBeforeSynchronize} gpuReadback={driverGpuReadback} correctness={driverCorrectnessProof} maxAbsoluteError={driverMaxAbsoluteError:G9} outputSha256={driverOutputSha256}");
        if (driverLaunchDiagnostic.Length != 0)
        {
            Console.WriteLine("driver.launch.diagnostic=" + driverLaunchDiagnostic.Replace(Environment.NewLine, " | "));
        }
        if (driverLoadSucceeded && (!driverLaunchSucceeded || !driverGpuReadback || !driverCorrectnessProof))
        {
            Console.Error.WriteLine("A loadable RTC artifact did not complete CUDA Driver owner-bound launch/readback correctness validation.");
            return 11;
        }

        var brokenSource = new CudaRtcProgramSource(
            "extern \"C\" __global__ void intentionally_broken( {\n",
            "intentional-failure.cu");
        CudaRtcCompilationResult failure = CudaRtcCompiler.Compile(brokenSource, options);
        if (failure.Success || failure.ResultCode != CudaRtcResultCode.Compilation || string.IsNullOrWhiteSpace(failure.Log))
        {
            Console.Error.WriteLine("Intentional compile failure did not preserve the expected NVRTC failure log.");
            return 5;
        }
        Console.WriteLine($"failure.success=False result={failure.ResultCode} logLength={failure.Log.Length}");
        string evidenceClassification = correctnessProof
            ? "local-toolkit-kernel-runtime-readback"
            : loadSucceeded ? "local-toolkit-compile-to-load" : "local-toolkit-compile-only-load-rejected";
        Console.WriteLine($"evidence.classification={evidenceClassification}; kernel-launch={launchSucceeded}; gpu-readback={gpuReadback}; correctness-proof={correctnessProof}");
        string driverEvidenceClassification = driverCorrectnessProof
            ? "local-toolkit-driver-kernel-runtime-readback"
            : driverLoadSucceeded ? "local-toolkit-driver-compile-to-load" : "local-toolkit-driver-compile-only-load-rejected";
        Console.WriteLine($"driver.evidence.classification={driverEvidenceClassification}; kernel-launch={driverLaunchSucceeded}; gpu-readback={driverGpuReadback}; correctness-proof={driverCorrectnessProof}");
        return 0;
    }
}
