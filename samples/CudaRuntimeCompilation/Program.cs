using JYPPX.CudaSharp;

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
try
{
    using CudaKernelLibrary library = CudaKernelLibrary.Load(ptx.ToArray());
    loadSucceeded = library.ContainsKernel("vector_add");
    if (!loadSucceeded)
    {
        loadDiagnostic = "PTX loaded, but vector_add was not found in the copied library inventory query.";
    }
}
catch (CudaException exception)
{
    loadDiagnostic = exception.Message;
}
Console.WriteLine($"load.attempted=True load.succeeded={loadSucceeded} classification=local-toolkit-compile-to-load");
if (loadDiagnostic.Length != 0)
{
    Console.WriteLine("load.diagnostic=" + loadDiagnostic.Replace(Environment.NewLine, " | "));
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
Console.WriteLine("evidence.classification=local-toolkit-compile-to-load; kernel-launch=False; gpu-readback=False; correctness-proof=False");
return 0;
