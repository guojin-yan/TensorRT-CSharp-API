using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

string repoRoot = ResolveRepoRoot(args);
string manifestsRoot = Path.Combine(repoRoot, "native", "manifests");
string templatesRoot = Path.Combine(repoRoot, "native", "templates");
string nativeOutput = Path.Combine(repoRoot, "native", "generated", "bridge_api_catalog.g.h");
string nativeDeclarationsOutput = Path.Combine(repoRoot, "native", "generated", "bridge_entrypoints.g.h");
string managedOutput = Path.Combine(repoRoot, "src", "JYPPX.Shared", "Generated", "GeneratedApiCatalog.g.cs");
string entryPointsOutput = Path.Combine(repoRoot, "src", "JYPPX.Shared", "Generated", "GeneratedEntryPointNames.g.cs");
string nativeMethodsOutput = Path.Combine(repoRoot, "src", "JYPPX.Shared", "Generated", "GeneratedNativeMethods.g.cs");
string tensorRtModuleOutput = Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "GeneratedTensorRtManifestNativeMethods.g.cs");
string cudaModuleOutput = Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "GeneratedCudaManifestNativeMethods.g.cs");
string commonPartialOutput = Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsCommon.Generated.g.cs");
string cudaCommonPartialOutput = Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "NativeMethodsCommon.Generated.g.cs");
string tensorRtPartialOutput = Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
string cudaPartialOutput = Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "NativeMethodsCuda.Generated.g.cs");
string tensorRtBridgeApiCommonOutput = Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeBridgeApi.Common.Generated.g.cs");
string cudaBridgeApiCommonOutput = Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "NativeBridgeApi.Common.Generated.g.cs");
string cudaApiGeneratedOutput = Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "NativeCudaApi.Generated.g.cs");
string tensorRtLineBindingsOutput = Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeBridgeApi.TensorRtBindings.Generated.g.cs");
string tensorRtLineHelpersOutput = Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeBridgeApi.TensorRtHelpers.Generated.g.cs");

List<ManifestApiRecord> apis = LoadApis(manifestsRoot);
string generatedOn = ShouldIncludeTimestamp()
    ? DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    : "deterministic";

string nativeTemplate = File.ReadAllText(Path.Combine(templatesRoot, "bridge_api_catalog.h.tpl"), Encoding.UTF8);
string nativeDeclarationsTemplate = File.ReadAllText(Path.Combine(templatesRoot, "bridge_entrypoints.h.tpl"), Encoding.UTF8);
string managedTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_api_catalog.cs.tpl"), Encoding.UTF8);
string entryPointsTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_entrypoint_names.cs.tpl"), Encoding.UTF8);
string nativeMethodsTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_native_methods.cs.tpl"), Encoding.UTF8);
string moduleNativeMethodsTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_module_native_methods.cs.tpl"), Encoding.UTF8);
string nativeBridgeApiCommonTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_native_bridge_api_common.cs.tpl"), Encoding.UTF8);
string nativeCudaApiTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_native_cuda_api.cs.tpl"), Encoding.UTF8);
string tensorRtLineBindingsTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_tensorrt_line_bindings.cs.tpl"), Encoding.UTF8);
string tensorRtLineHelpersTemplate = File.ReadAllText(Path.Combine(templatesRoot, "generated_tensorrt_line_helpers.cs.tpl"), Encoding.UTF8);

string nativeRows = string.Join(
    Environment.NewLine,
    apis.Select(api =>
        $"#define JYPPX_API_{SanitizeForMacro(api.Id)} \"{api.EntryPoint}\" // module={api.Module}; line={api.VersionLine}; ownership={api.Ownership}; manualOverride={(api.ManualOverride ? "true" : "false")}"));

string nativeDeclarationRows = string.Join(
    Environment.NewLine + Environment.NewLine,
    apis.Select(api =>
        $"// generated from manifest id={api.Id}{Environment.NewLine}JYPPX_C_API({api.ReturnType}) {api.EntryPoint}({BuildNativeParameterList(api)});"));

string managedRows = string.Join(
    "," + Environment.NewLine,
    apis.Select(api =>
        $"            new GeneratedApiDefinition(\"{api.Id}\", \"{api.Module}\", \"{api.VersionLine}\", \"{api.EntryPoint}\", \"{Escape(api.Ownership)}\", {api.ManualOverride.ToString().ToLowerInvariant()}, \"{Escape(api.VersionGuard)}\", \"{Escape(BuildParameterSummary(api.Parameters))}\")"));

string entryPointRows = string.Join(
    Environment.NewLine,
    apis.Select(api => $"    public const string {SanitizeForIdentifier(api.Id)} = \"{api.EntryPoint}\";"));

List<ManifestApiRecord> sharedNativeMethodApis = apis
    .Where(api => string.Equals(api.Module, "common", StringComparison.OrdinalIgnoreCase))
    .ToList();

string nativeMethodRows = string.Join(
    Environment.NewLine + Environment.NewLine,
    sharedNativeMethodApis.Select(api =>
        $"    [DllImport(BridgeConstants.NativeBridgeLibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]{Environment.NewLine}    internal static extern {api.ReturnType} {SanitizeForIdentifier(api.Id)}({BuildManagedParameterList(api)});"));

Directory.CreateDirectory(Path.GetDirectoryName(nativeOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(nativeDeclarationsOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(managedOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(entryPointsOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(nativeMethodsOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(tensorRtModuleOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(cudaModuleOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(commonPartialOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(cudaCommonPartialOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(tensorRtPartialOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(cudaPartialOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(tensorRtBridgeApiCommonOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(cudaBridgeApiCommonOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(cudaApiGeneratedOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(tensorRtLineBindingsOutput)!);
Directory.CreateDirectory(Path.GetDirectoryName(tensorRtLineHelpersOutput)!);

WriteAllTextAtomic(
    nativeOutput,
    nativeTemplate.Replace("{{GeneratedOn}}", generatedOn).Replace("{{ApiRows}}", nativeRows),
    new UTF8Encoding(false));

WriteAllTextAtomic(
    nativeDeclarationsOutput,
    nativeDeclarationsTemplate.Replace("{{GeneratedOn}}", generatedOn).Replace("{{ApiDeclarationRows}}", nativeDeclarationRows),
    new UTF8Encoding(false));

WriteAllTextAtomic(
    managedOutput,
    managedTemplate.Replace("{{GeneratedOn}}", generatedOn).Replace("{{ApiRows}}", managedRows),
    new UTF8Encoding(false));

WriteAllTextAtomic(
    entryPointsOutput,
    entryPointsTemplate.Replace("{{GeneratedOn}}", generatedOn).Replace("{{EntryPointRows}}", entryPointRows),
    new UTF8Encoding(false));

WriteAllTextAtomic(
    nativeMethodsOutput,
    nativeMethodsTemplate.Replace("{{GeneratedOn}}", generatedOn).Replace("{{NativeMethodRows}}", nativeMethodRows),
    new UTF8Encoding(false));

WriteModuleNativeMethods(
    outputPath: tensorRtModuleOutput,
    generatedOn: generatedOn,
    templateText: moduleNativeMethodsTemplate,
    targetNamespace: "JYPPX.TensorRtSharp.Internal.Interop",
    className: "GeneratedTensorRtManifestNativeMethods",
    extraUsings: "using JYPPX.CudaSharp.Internal.Handles;" + Environment.NewLine + "using JYPPX.TensorRtSharp.Internal.Handles;",
    apis: apis.Where(api => string.Equals(api.Module, "tensorrt", StringComparison.OrdinalIgnoreCase)).ToList());

WriteModuleNativeMethods(
    outputPath: cudaModuleOutput,
    generatedOn: generatedOn,
    templateText: moduleNativeMethodsTemplate,
    targetNamespace: "JYPPX.CudaSharp.Internal.Interop",
    className: "GeneratedCudaManifestNativeMethods",
    extraUsings: "using JYPPX.CudaSharp.Internal.Handles;",
    apis: apis.Where(api => string.Equals(api.Module, "cuda", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeMethodsPartial(
    outputPath: commonPartialOutput,
    generatedOn: generatedOn,
    templateText: moduleNativeMethodsTemplate,
    targetNamespace: "JYPPX.TensorRtSharp.Internal.Interop",
    className: "NativeMethodsCommon",
    extraUsings: string.Empty,
    existingSourcePath: Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeMethodsCommon.cs"),
    apis: apis.Where(api => string.Equals(api.Module, "common", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeMethodsPartial(
    outputPath: cudaCommonPartialOutput,
    generatedOn: generatedOn,
    templateText: moduleNativeMethodsTemplate,
    targetNamespace: "JYPPX.CudaSharp.Internal.Interop",
    className: "NativeMethodsCommon",
    extraUsings: string.Empty,
    existingSourcePath: Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeMethodsCommon.cs"),
    apis: apis.Where(api => string.Equals(api.Module, "common", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeMethodsPartial(
    outputPath: tensorRtPartialOutput,
    generatedOn: generatedOn,
    templateText: moduleNativeMethodsTemplate,
    targetNamespace: "JYPPX.TensorRtSharp.Internal.Interop",
    className: "NativeMethodsTensorRt",
    extraUsings: "using JYPPX.CudaSharp.Internal.Handles;" + Environment.NewLine + "using JYPPX.TensorRtSharp.Internal.Handles;",
    existingSourcePath: Path.Combine(repoRoot, "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeMethodsTensorRt.cs"),
    apis: apis.Where(api => string.Equals(api.Module, "tensorrt", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeMethodsPartial(
    outputPath: cudaPartialOutput,
    generatedOn: generatedOn,
    templateText: moduleNativeMethodsTemplate,
    targetNamespace: "JYPPX.CudaSharp.Internal.Interop",
    className: "NativeMethodsCuda",
    extraUsings: "using JYPPX.CudaSharp.Internal.Handles;",
    existingSourcePath: Path.Combine(repoRoot, "src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeMethodsCuda.cs"),
    apis: apis.Where(api => string.Equals(api.Module, "cuda", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeBridgeApiCommonWrapper(
    outputPath: tensorRtBridgeApiCommonOutput,
    generatedOn: generatedOn,
    templateText: nativeBridgeApiCommonTemplate,
    targetNamespace: "JYPPX.TensorRtSharp.Internal.Interop",
    statusHelperClass: "NativeStatus",
    includeCapabilityInfo: true,
    apis: apis.Where(api => string.Equals(api.Module, "common", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeBridgeApiCommonWrapper(
    outputPath: cudaBridgeApiCommonOutput,
    generatedOn: generatedOn,
    templateText: nativeBridgeApiCommonTemplate,
    targetNamespace: "JYPPX.CudaSharp.Internal.Interop",
    statusHelperClass: "CudaNativeStatus",
    includeCapabilityInfo: true,
    apis: apis.Where(api => string.Equals(api.Module, "common", StringComparison.OrdinalIgnoreCase)).ToList());

WriteNativeCudaApiWrapper(
    outputPath: cudaApiGeneratedOutput,
    generatedOn: generatedOn,
    templateText: nativeCudaApiTemplate,
    apis: apis.Where(api => string.Equals(api.Module, "cuda", StringComparison.OrdinalIgnoreCase)).ToList());

WriteTensorRtLineBindings(
    outputPath: tensorRtLineBindingsOutput,
    generatedOn: generatedOn,
    templateText: tensorRtLineBindingsTemplate,
    apis: apis.Where(api => string.Equals(api.Module, "tensorrt", StringComparison.OrdinalIgnoreCase)).ToList());

WriteTensorRtLineHelpers(
    outputPath: tensorRtLineHelpersOutput,
    generatedOn: generatedOn,
    templateText: tensorRtLineHelpersTemplate,
    apis: apis.Where(api => string.Equals(api.Module, "tensorrt", StringComparison.OrdinalIgnoreCase)).ToList());

Console.WriteLine($"Generated {apis.Count} API records.");
Console.WriteLine(nativeOutput);
Console.WriteLine(nativeDeclarationsOutput);
Console.WriteLine(managedOutput);
Console.WriteLine(entryPointsOutput);
Console.WriteLine(nativeMethodsOutput);
Console.WriteLine(tensorRtModuleOutput);
Console.WriteLine(cudaModuleOutput);
Console.WriteLine(commonPartialOutput);
Console.WriteLine(cudaCommonPartialOutput);
Console.WriteLine(tensorRtPartialOutput);
Console.WriteLine(cudaPartialOutput);
Console.WriteLine(tensorRtBridgeApiCommonOutput);
Console.WriteLine(cudaBridgeApiCommonOutput);
Console.WriteLine(cudaApiGeneratedOutput);
Console.WriteLine(tensorRtLineBindingsOutput);
Console.WriteLine(tensorRtLineHelpersOutput);

static void WriteAllTextAtomic(string outputPath, string contents, Encoding encoding)
{
    string directory = Path.GetDirectoryName(outputPath) ?? Directory.GetCurrentDirectory();
    Directory.CreateDirectory(directory);

    string tempPath = Path.Combine(directory, $"{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.tmp");
    string backupPath = Path.Combine(directory, $"{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.bak");
    File.WriteAllText(tempPath, contents, encoding);

    try
    {
        try
        {
            File.Move(tempPath, outputPath, overwrite: true);
        }
        catch (IOException)
        {
            if (File.Exists(outputPath))
            {
                File.Move(outputPath, backupPath, overwrite: true);
            }

            File.Move(tempPath, outputPath, overwrite: true);

            if (File.Exists(backupPath))
            {
                File.Delete(backupPath);
            }
        }
    }
    finally
    {
        if (File.Exists(tempPath))
        {
            File.Delete(tempPath);
        }
    }
}
static string ResolveRepoRoot(string[] args)
{
    for (int i = 0; i < args.Length; i++)
    {
        if (string.Equals(args[i], "--repo-root", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
        {
            return Path.GetFullPath(args[i + 1]);
        }
    }

    return Directory.GetCurrentDirectory();
}

static bool ShouldIncludeTimestamp()
{
    string? value = Environment.GetEnvironmentVariable("JYPPX_BINDING_GENERATOR_INCLUDE_TIMESTAMP");
    return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase)
        || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
}

static List<ManifestApiRecord> LoadApis(string manifestsRoot)
{
    List<ManifestApiRecord> records = new List<ManifestApiRecord>();
    JsonSerializerOptions options = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    };

    foreach (string manifestPath in Directory.EnumerateFiles(manifestsRoot, "*.manifest.json", SearchOption.AllDirectories).OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
    {
        ManifestDocument? document = JsonSerializer.Deserialize<ManifestDocument>(File.ReadAllText(manifestPath, Encoding.UTF8), options);
        if (document == null)
        {
            throw new InvalidOperationException($"Unable to deserialize manifest: {manifestPath}");
        }

        ValidateManifest(manifestPath, document);

        foreach (ManifestApi api in document.Apis)
        {
            List<ManifestParameterRecord> parameters = api.Parameters.Select(parameter => new ManifestParameterRecord(
                Name: parameter.Name,
                NativeType: parameter.Type,
                Direction: parameter.Direction,
                Ownership: parameter.Ownership ?? string.Empty,
                ManagedType: parameter.ManagedType ?? string.Empty,
                ModuleManagedType: parameter.ModuleManagedType ?? string.Empty)).ToList();

            records.Add(new ManifestApiRecord(
                Id: api.Id,
                Module: document.Module,
                VersionLine: document.VersionLine,
                EntryPoint: api.EntryPoint,
                ReturnType: string.IsNullOrWhiteSpace(api.ReturnType) ? "BridgeStatusCode" : api.ReturnType,
                Ownership: api.Ownership,
                ManualOverride: api.ManualOverride,
                VersionGuard: api.VersionGuard ?? string.Empty,
                WrapperKind: api.WrapperKind ?? string.Empty,
                BindingRole: api.BindingRole ?? string.Empty,
                Parameters: parameters));
        }
    }

    return records.OrderBy(record => record.Module, StringComparer.OrdinalIgnoreCase)
        .ThenBy(record => record.VersionLine, StringComparer.OrdinalIgnoreCase)
        .ThenBy(record => record.Id, StringComparer.OrdinalIgnoreCase)
        .ToList();
}

static void ValidateManifest(string path, ManifestDocument document)
{
    if (string.IsNullOrWhiteSpace(document.Module))
    {
        throw new InvalidOperationException($"Manifest '{path}' is missing 'module'.");
    }

    if (string.IsNullOrWhiteSpace(document.VersionLine))
    {
        throw new InvalidOperationException($"Manifest '{path}' is missing 'versionLine'.");
    }

    if (document.Apis == null || document.Apis.Count == 0)
    {
        throw new InvalidOperationException($"Manifest '{path}' must contain at least one API entry.");
    }

    foreach (ManifestApi api in document.Apis)
    {
        if (string.IsNullOrWhiteSpace(api.Id) ||
            string.IsNullOrWhiteSpace(api.EntryPoint) ||
            string.IsNullOrWhiteSpace(api.ReturnType) ||
            string.IsNullOrWhiteSpace(api.Ownership) ||
            api.Parameters == null)
        {
            throw new InvalidOperationException($"Manifest '{path}' contains an API entry with missing required fields.");
        }

        foreach (ManifestParameter parameter in api.Parameters)
        {
            if (string.IsNullOrWhiteSpace(parameter.Name) ||
                string.IsNullOrWhiteSpace(parameter.Type) ||
                string.IsNullOrWhiteSpace(parameter.Direction))
            {
                throw new InvalidOperationException($"Manifest '{path}' contains a parameter with missing required fields.");
            }
        }
    }
}

static string SanitizeForMacro(string value)
{
    StringBuilder builder = new StringBuilder(value.Length);
    foreach (char character in value)
    {
        builder.Append(char.IsLetterOrDigit(character) ? char.ToUpperInvariant(character) : '_');
    }

    return builder.ToString();
}

static string SanitizeForIdentifier(string value)
{
    StringBuilder builder = new StringBuilder(value.Length);
    bool upperNext = true;
    foreach (char character in value)
    {
        if (char.IsLetterOrDigit(character))
        {
            builder.Append(upperNext ? char.ToUpperInvariant(character) : character);
            upperNext = false;
        }
        else
        {
            upperNext = true;
        }
    }

    return builder.Length == 0 ? "UnknownEntryPoint" : builder.ToString();
}

static string Escape(string value)
{
    return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
}

static string BuildNativeParameterList(ManifestApiRecord api)
{
    if (api.Parameters.Count == 0)
    {
        return "void";
    }

    return string.Join(", ", api.Parameters.Select(parameter => $"{parameter.NativeType} {parameter.Name}"));
}

static string BuildManagedParameterList(ManifestApiRecord api)
{
    return BuildManagedParameterListCore(api.Parameters, moduleSpecific: false);
}

static string BuildModuleManagedParameterList(ManifestApiRecord api)
{
    return BuildManagedParameterListCore(api.Parameters, moduleSpecific: true);
}

static string BuildManagedParameterListCore(IReadOnlyList<ManifestParameterRecord> parameters, bool moduleSpecific)
{
    if (parameters.Count == 0)
    {
        return string.Empty;
    }

    string[] parts = parameters
        .Select(parameter =>
        {
            string managedType = GetManagedType(parameter, moduleSpecific);
            return $"{managedType} {SanitizeParameterName(parameter.Name)}";
        })
        .ToArray();

    return string.Join(", ", parts);
}

static string GetManagedType(ManifestParameterRecord parameter, bool moduleSpecific)
{
    string overrideType = moduleSpecific ? parameter.ModuleManagedType : parameter.ManagedType;
    if (!string.IsNullOrWhiteSpace(overrideType))
    {
        return overrideType;
    }

    return parameter.NativeType switch
    {
        "JYPPX_StatusCode" => "BridgeStatusCode",
        "JYPPX_ErrorCategory" => "BridgeErrorCategory",
        "JYPPX_Boolean" => "int",
        "JYPPX_Boolean*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out int",
        "uint32_t" => "uint",
        "uint32_t*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out uint",
        "uint64_t" => "ulong",
        "uint64_t*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out ulong",
        "uintptr_t" => "UIntPtr",
        "uintptr_t*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out UIntPtr",
        "int32_t" => "int",
        "int32_t*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out int",
        "int64_t" => "long",
        "int64_t*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out long",
        "float" => "float",
        "float*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out float",
        "double" => "double",
        "double*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out double",
        "size_t" => "UIntPtr",
        "size_t*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out UIntPtr",
        "UIntPtr*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out UIntPtr",
        "JYPPX_CudaStream**" => moduleSpecific ? "out SafeCudaStreamHandle" : "out IntPtr",
        "JYPPX_CudaEvent**" => moduleSpecific ? "out SafeCudaEventHandle" : "out IntPtr",
        "JYPPX_CudaMemory**" => moduleSpecific ? "out SafeCudaMemoryHandle" : "out IntPtr",
        "JYPPX_CudaPinnedMemory**" => moduleSpecific ? "out SafeCudaPinnedMemoryHandle" : "out IntPtr",
        "JYPPX_CudaPitchedMemory**" => moduleSpecific ? "out SafeCudaPitchedMemoryHandle" : "out IntPtr",
        "JYPPX_CudaGraph**" => moduleSpecific ? "out SafeCudaGraphHandle" : "out IntPtr",
        "JYPPX_CudaGraphExec**" => moduleSpecific ? "out SafeCudaGraphExecHandle" : "out IntPtr",
        "JYPPX_CudaGraphMemoryAllocation**" => moduleSpecific ? "out SafeCudaGraphMemoryAllocationHandle" : "out IntPtr",
        "JYPPX_CudaGraphConditionalHandle**" => moduleSpecific ? "out SafeCudaGraphConditionalHandleHandle" : "out IntPtr",
        "JYPPX_CudaGraphConditionalNode**" => moduleSpecific ? "out SafeCudaGraphConditionalNodeHandle" : "out IntPtr",
        "JYPPX_CudaArray**" => moduleSpecific ? "out SafeCudaArrayHandle" : "out IntPtr",
        "JYPPX_CudaMipmappedArray**" => moduleSpecific ? "out SafeCudaMipmappedArrayHandle" : "out IntPtr",
        "JYPPX_CudaTextureObject**" => moduleSpecific ? "out SafeCudaTextureObjectHandle" : "out IntPtr",
        "JYPPX_CudaSurfaceObject**" => moduleSpecific ? "out SafeCudaSurfaceObjectHandle" : "out IntPtr",
        "JYPPX_CudaKernelLibrary**" => moduleSpecific ? "out SafeCudaKernelLibraryHandle" : "out IntPtr",
        "JYPPX_CudaKernelLaunch**" => moduleSpecific ? "out SafeCudaKernelLaunchHandle" : "out IntPtr",
        "JYPPX_CudaDriverModule**" => moduleSpecific ? "out SafeCudaDriverModuleHandle" : "out IntPtr",
        "JYPPX_CudaDriverKernelLaunch**" => moduleSpecific ? "out SafeCudaDriverKernelLaunchHandle" : "out IntPtr",
        "JYPPX_CudaRtcProgram**" => moduleSpecific ? "out SafeCudaRtcProgramHandle" : "out IntPtr",
        "JYPPX_CudaExecutionContext**" => moduleSpecific ? "out SafeCudaExecutionContextHandle" : "out IntPtr",
        "JYPPX_CudaStream*" => moduleSpecific ? "SafeCudaStreamHandle" : "IntPtr",
        "JYPPX_CudaEvent*" => moduleSpecific ? "SafeCudaEventHandle" : "IntPtr",
        "JYPPX_CudaMemory*" => moduleSpecific ? "SafeCudaMemoryHandle" : "IntPtr",
        "JYPPX_CudaPinnedMemory*" => moduleSpecific ? "SafeCudaPinnedMemoryHandle" : "IntPtr",
        "JYPPX_CudaPitchedMemory*" => moduleSpecific ? "SafeCudaPitchedMemoryHandle" : "IntPtr",
        "JYPPX_CudaGraph*" => moduleSpecific ? "SafeCudaGraphHandle" : "IntPtr",
        "JYPPX_CudaGraphExec*" => moduleSpecific ? "SafeCudaGraphExecHandle" : "IntPtr",
        "JYPPX_CudaGraphMemoryAllocation*" => moduleSpecific ? "SafeCudaGraphMemoryAllocationHandle" : "IntPtr",
        "JYPPX_CudaGraphConditionalHandle*" => moduleSpecific ? "SafeCudaGraphConditionalHandleHandle" : "IntPtr",
        "JYPPX_CudaGraphConditionalNode*" => moduleSpecific ? "SafeCudaGraphConditionalNodeHandle" : "IntPtr",
        "JYPPX_CudaArray*" => moduleSpecific ? "SafeCudaArrayHandle" : "IntPtr",
        "JYPPX_CudaMipmappedArray*" => moduleSpecific ? "SafeCudaMipmappedArrayHandle" : "IntPtr",
        "JYPPX_CudaTextureObject*" => moduleSpecific ? "SafeCudaTextureObjectHandle" : "IntPtr",
        "JYPPX_CudaSurfaceObject*" => moduleSpecific ? "SafeCudaSurfaceObjectHandle" : "IntPtr",
        "JYPPX_CudaKernelLibrary*" => moduleSpecific ? "SafeCudaKernelLibraryHandle" : "IntPtr",
        "JYPPX_CudaKernelLaunch*" => moduleSpecific ? "SafeCudaKernelLaunchHandle" : "IntPtr",
        "JYPPX_CudaDriverModule*" => moduleSpecific ? "SafeCudaDriverModuleHandle" : "IntPtr",
        "JYPPX_CudaDriverKernelLaunch*" => moduleSpecific ? "SafeCudaDriverKernelLaunchHandle" : "IntPtr",
        "JYPPX_CudaRtcProgram*" => moduleSpecific ? "SafeCudaRtcProgramHandle" : "IntPtr",
        "JYPPX_CudaExecutionContext*" => moduleSpecific ? "SafeCudaExecutionContextHandle" : "IntPtr",
        "JYPPX_TensorRtRuntime**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtHostMemory**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtBuilder**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtBuilderConfig**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtNetworkDefinition**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtCudaEngine**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtExecutionContext**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtOnnxParser**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtOnnxParserRefitter**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtOnnxConfig**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtOptimizationProfile**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtEngineInspector**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtTimingCache**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtTensor**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtLayer**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtRefitter**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtLoop**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtIfConditional**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtSerializationConfig**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtRuntimeConfig**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtAttention**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtLogger**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtProgressMonitor**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtAllocatorOwner**" => moduleSpecific ? "out SafeTensorRtObjectHandle" : "out IntPtr",
        "JYPPX_TensorRtLogger*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtProgressMonitor*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtAllocatorOwner*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtBuilder*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtBuilderConfig*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtNetworkDefinition*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtRuntime*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtHostMemory*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtCudaEngine*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtExecutionContext*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtOnnxParser*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtOnnxParserRefitter*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtOnnxConfig*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtOptimizationProfile*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtEngineInspector*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtTimingCache*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtTensor*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtLayer*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtRefitter*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtLoop*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtIfConditional*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtSerializationConfig*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtRuntimeConfig*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtAttention*" => moduleSpecific ? "SafeTensorRtObjectHandle" : "IntPtr",
        "JYPPX_TensorRtAdapterInfo*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtAdapterInfo" : "out IntPtr",
        "JYPPX_TensorRtTensorInfo*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtTensorInfo" : "out IntPtr",
        "JYPPX_TensorRtDims*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtDims" : "out IntPtr",
        "JYPPX_TensorRtWeightsInfo*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtWeightsInfo" : "out IntPtr",
        "JYPPX_TensorRtAllocatorOwnerDiagnosticInfo*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtAllocatorOwnerDiagnosticInfo" : "out IntPtr",
        "JYPPX_TensorRtAllocatorOwnerStateInfo*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtAllocatorOwnerStateInfo" : "out IntPtr",
        "JYPPX_TensorRtExecutionContextCallbackStateInfo*" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => moduleSpecific ? "out NativeTensorRtExecutionContextCallbackStateInfo" : "out IntPtr",
        "JYPPX_TensorRtDims*" => moduleSpecific ? "ref NativeTensorRtDims" : "IntPtr",
        "const JYPPX_TensorRtDims*" => moduleSpecific ? "ref NativeTensorRtDims" : "IntPtr",
        "const void*" => "IntPtr",
        "void*" => "IntPtr",
        "void**" when string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out IntPtr",
        _ when parameter.NativeType.EndsWith("**", StringComparison.Ordinal) && string.Equals(parameter.Direction, "out", StringComparison.OrdinalIgnoreCase) => "out IntPtr",
        _ when parameter.NativeType.EndsWith("*", StringComparison.Ordinal) => "IntPtr",
        _ => "IntPtr"
    };
}

static string BuildParameterSummary(IReadOnlyList<ManifestParameterRecord> parameters)
{
    return string.Join(", ", parameters.Select(parameter =>
    {
        string ownership = string.IsNullOrWhiteSpace(parameter.Ownership) ? "n/a" : parameter.Ownership;
        return $"{parameter.Direction}:{parameter.Name}:{parameter.NativeType}:{ownership}";
    }));
}

static void WriteModuleNativeMethods(
    string outputPath,
    string generatedOn,
    string templateText,
    string targetNamespace,
    string className,
    string extraUsings,
    List<ManifestApiRecord> apis)
{
    string rows = string.Join(
        Environment.NewLine + Environment.NewLine,
        apis.Select(api =>
            $"    [DllImport(BridgeConstants.NativeBridgeLibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]{Environment.NewLine}    internal static extern {api.ReturnType} {api.EntryPoint}({BuildModuleManagedParameterList(api)});"));

    WriteAllTextAtomic(
        outputPath,
        templateText
            .Replace("{{GeneratedOn}}", generatedOn)
            .Replace("{{Namespace}}", targetNamespace)
            .Replace("{{ClassName}}", className)
            .Replace("{{ExtraUsings}}", extraUsings)
            .Replace("{{NativeMethodRows}}", rows),
        new UTF8Encoding(false));
}

static void WriteNativeMethodsPartial(
    string outputPath,
    string generatedOn,
    string templateText,
    string targetNamespace,
    string className,
    string extraUsings,
    string existingSourcePath,
    List<ManifestApiRecord> apis)
{
    string existingContent = File.Exists(existingSourcePath) ? File.ReadAllText(existingSourcePath, Encoding.UTF8) : string.Empty;
    List<ManifestApiRecord> missingApis = apis
        .Where(api => !existingContent.Contains(api.EntryPoint + "(", StringComparison.Ordinal))
        .ToList();

    string rows = string.Join(
        Environment.NewLine + Environment.NewLine,
        missingApis.Select(api =>
            $"    [DllImport(BridgeConstants.NativeBridgeLibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]{Environment.NewLine}    internal static extern {api.ReturnType} {api.EntryPoint}({BuildModuleManagedParameterList(api)});"));

    if (string.IsNullOrWhiteSpace(rows))
    {
        rows = "    // No missing manifest-driven interop declarations for this module in the current generation pass.";
    }

    WriteAllTextAtomic(
        outputPath,
        templateText
            .Replace("{{GeneratedOn}}", generatedOn)
            .Replace("{{Namespace}}", targetNamespace)
            .Replace("{{ClassName}}", className)
            .Replace("{{ExtraUsings}}", extraUsings)
            .Replace("{{NativeMethodRows}}", rows),
        new UTF8Encoding(false));
}

static void WriteNativeBridgeApiCommonWrapper(
    string outputPath,
    string generatedOn,
    string templateText,
    string targetNamespace,
    string statusHelperClass,
    bool includeCapabilityInfo,
    List<ManifestApiRecord> apis)
{
    HashSet<string> entryPoints = apis
        .Select(api => api.EntryPoint)
        .ToHashSet(StringComparer.Ordinal);

    List<string> methodRows = new List<string>();

    if (entryPoints.Contains("jyppx_common_get_build_info"))
    {
        methodRows.Add(
@"    public static NativeBuildInfo GetBuildInfo()
    {
        " + statusHelperClass + @"." + @"ThrowIfFailed(NativeMethodsCommon.jyppx_common_get_build_info(out NativeBuildInfo value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_common_query_runtime_info"))
    {
        methodRows.Add(
@"    public static NativeRuntimeInfo GetRuntimeInfo()
    {
        " + statusHelperClass + @"." + @"ThrowIfFailed(NativeMethodsCommon.jyppx_common_query_runtime_info(out NativeRuntimeInfo value));
        return value;
    }");
    }

    if (includeCapabilityInfo && entryPoints.Contains("jyppx_common_query_capability_info"))
    {
        methodRows.Add(
@"    public static NativeCapabilityInfo GetCapabilityInfo()
    {
        " + statusHelperClass + @"." + @"ThrowIfFailed(NativeMethodsCommon.jyppx_common_query_capability_info(out NativeCapabilityInfo value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_common_get_last_error_message"))
    {
        methodRows.Add(
@"    public static string GetLastErrorMessageOrFallback(string fallback)
    {
        byte[] buffer = new byte[512];
        BridgeStatusCode status = NativeMethodsCommon.jyppx_common_get_last_error_message(buffer, (UIntPtr)buffer.Length, out UIntPtr requiredSize);
        if (status == BridgeStatusCode.BufferTooSmall && requiredSize.ToUInt64() > (ulong)buffer.Length)
        {
            buffer = new byte[checked((int)requiredSize.ToUInt64())];
            status = NativeMethodsCommon.jyppx_common_get_last_error_message(buffer, requiredSize, out requiredSize);
        }

        if (status != BridgeStatusCode.Ok)
        {
            return fallback;
        }

        int terminatorIndex = Array.IndexOf(buffer, (byte)0);
        int length = terminatorIndex >= 0 ? terminatorIndex : buffer.Length;
        return length == 0 ? fallback : Encoding.UTF8.GetString(buffer, 0, length);
    }");
    }

    if (entryPoints.Contains("jyppx_common_get_last_error_category"))
    {
        methodRows.Add(
@"    public static BridgeErrorCategory GetLastErrorCategory()
    {
        return NativeMethodsCommon.jyppx_common_get_last_error_category();
    }");
    }

    if (entryPoints.Contains("jyppx_common_clear_last_error"))
    {
        methodRows.Add(
@"    public static void ClearLastError()
    {
        NativeMethodsCommon.jyppx_common_clear_last_error();
    }");
    }

    string rows = string.Join(Environment.NewLine + Environment.NewLine, methodRows);
    if (string.IsNullOrWhiteSpace(rows))
    {
        rows = "    // No common bridge helper methods were generated for this namespace.";
    }

    WriteAllTextAtomic(
        outputPath,
        templateText
            .Replace("{{GeneratedOn}}", generatedOn)
            .Replace("{{Namespace}}", targetNamespace)
            .Replace("{{MethodRows}}", rows),
        new UTF8Encoding(false));
}

static void WriteNativeCudaApiWrapper(
    string outputPath,
    string generatedOn,
    string templateText,
    List<ManifestApiRecord> apis)
{
    HashSet<string> entryPoints = apis.Select(api => api.EntryPoint).ToHashSet(StringComparer.Ordinal);
    Dictionary<string, ManifestApiRecord> wrapperKindMap = apis
        .Where(api => !string.IsNullOrWhiteSpace(api.WrapperKind))
        .ToDictionary(api => api.WrapperKind, api => api, StringComparer.Ordinal);

    List<string> descriptorRows = new List<string>();
    List<string> methodRows = new List<string>();

    if (entryPoints.Contains("jyppx_cuda_query_runtime_info"))
    {
        methodRows.Add(
@"    public static NativeCudaRuntimeInfo GetRuntimeInfo()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_query_runtime_info(out NativeCudaRuntimeInfo value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_runtime_version"))
    {
        methodRows.Add(
@"    public static int GetRuntimeVersion()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_runtime_version(out int version));
        return version;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_driver_version"))
    {
        methodRows.Add(
@"    public static int GetDriverVersion()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_driver_version(out int version));
        return version;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_device_count"))
    {
        methodRows.Add(
@"    public static int GetDeviceCount()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_device_count(out int count));
        return count;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_device_info"))
    {
        methodRows.Add(
@"    public static NativeCudaDeviceInfo GetDeviceInfo(int ordinal)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_device_info(ordinal, out NativeCudaDeviceInfo value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_current_device"))
    {
        methodRows.Add(
@"    public static int GetCurrentDevice()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_current_device(out int device));
        return device;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_set_device"))
    {
        methodRows.Add(
@"    public static void SetDevice(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_set_device(device));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_attribute"))
    {
        methodRows.Add(
@"    public static int GetDeviceAttribute(int device, int attribute)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_attribute(device, attribute, out int value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_limit"))
    {
        methodRows.Add(
@"    public static ulong GetDeviceLimit(int limit)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_limit(limit, out UIntPtr value));
        return value.ToUInt64();
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_set_limit"))
    {
        methodRows.Add(
@"    public static void SetDeviceLimit(int limit, ulong value)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_set_limit(limit, (UIntPtr)value));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_cache_config"))
    {
        methodRows.Add(
@"    public static int GetDeviceCacheConfig()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_cache_config(out int config));
        return config;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_set_cache_config"))
    {
        methodRows.Add(
@"    public static void SetDeviceCacheConfig(int config)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_set_cache_config(config));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_shared_memory_config"))
    {
        methodRows.Add(
@"    public static int GetSharedMemoryConfig()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_shared_memory_config(out int config));
        return config;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_set_shared_memory_config"))
    {
        methodRows.Add(
@"    public static void SetSharedMemoryConfig(int config)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_set_shared_memory_config(config));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_can_access_peer"))
    {
        methodRows.Add(
@"    public static bool CanDeviceAccessPeer(int device, int peerDevice)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_can_access_peer(device, peerDevice, out int canAccess));
        return canAccess != 0;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_p2p_attribute"))
    {
        methodRows.Add(
@"    public static int GetDeviceP2PAttribute(int attribute, int sourceDevice, int destinationDevice)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_p2p_attribute(attribute, sourceDevice, destinationDevice, out int value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_enable_peer_access"))
    {
        methodRows.Add(
@"    public static void EnablePeerAccess(int peerDevice, uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_enable_peer_access(peerDevice, flags));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_disable_peer_access"))
    {
        methodRows.Add(
@"    public static void DisablePeerAccess(int peerDevice)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_disable_peer_access(peerDevice));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_device_flags"))
    {
        methodRows.Add(
@"    public static uint GetDeviceFlags()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_device_flags(out uint flags));
        return flags;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_set_device_flags"))
    {
        methodRows.Add(
@"    public static void SetDeviceFlags(uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_set_device_flags(flags));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_get_memory_info"))
    {
        methodRows.Add(
@"    public static NativeCudaMemoryInfo GetMemoryInfo()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_memory_info(out NativeCudaMemoryInfo value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_pool_create"))
    {
        methodRows.Add(
@"    public static ulong CreateMemoryPoolHandle(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_create(device, out ulong poolHandle));
        return poolHandle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_pool_destroy"))
    {
        methodRows.Add(
@"    public static void DestroyMemoryPoolHandle(ulong poolHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_destroy(poolHandle));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_default_memory_pool"))
    {
        methodRows.Add(
@"    public static ulong GetDefaultMemoryPoolHandle(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_default_memory_pool(device, out ulong poolHandle));
        return poolHandle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_get_current_memory_pool"))
    {
        methodRows.Add(
@"    public static ulong GetCurrentMemoryPoolHandle(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_current_memory_pool(device, out ulong poolHandle));
        return poolHandle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_set_current_memory_pool"))
    {
        methodRows.Add(
@"    public static void SetCurrentMemoryPoolHandle(int device, ulong poolHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_set_current_memory_pool(device, poolHandle));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_pool_get_attribute"))
    {
        methodRows.Add(
@"    public static long GetMemoryPoolAttribute(ulong poolHandle, int attribute)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_get_attribute(poolHandle, attribute, out long value));
        return value;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_pool_set_attribute"))
    {
        methodRows.Add(
@"    public static void SetMemoryPoolAttribute(ulong poolHandle, int attribute, long value)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_set_attribute(poolHandle, attribute, value));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_pool_trim_to"))
    {
        methodRows.Add(
@"    public static void TrimMemoryPoolTo(ulong poolHandle, ulong minBytesToKeep)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_trim_to(poolHandle, (UIntPtr)minBytesToKeep));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_synchronize"))
    {
        methodRows.Add(
@"    public static void SynchronizeDevice()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_synchronize());
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_device_reset"))
    {
        methodRows.Add(
@"    public static void ResetDevice()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_reset());
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_create"))
    {
        methodRows.Add(
@"    public static SafeCudaStreamHandle CreateStream()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_create(0, out SafeCudaStreamHandle handle));
        return handle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_get_priority_range"))
    {
        methodRows.Add(
@"    public static void GetStreamPriorityRange(out int leastPriority, out int greatestPriority)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_priority_range(out leastPriority, out greatestPriority));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_create_with_priority"))
    {
        methodRows.Add(
@"    public static SafeCudaStreamHandle CreateStreamWithPriority(uint flags, int priority)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_create_with_priority(flags, priority, out SafeCudaStreamHandle handle));
        return handle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_get_priority"))
    {
        methodRows.Add(
@"    public static int GetStreamPriority(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_priority(stream, out int priority));
        return priority;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_get_id"))
    {
        methodRows.Add(
@"    public static ulong GetStreamId(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_id(stream, out ulong streamId));
        return streamId;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_get_device"))
    {
        methodRows.Add(
@"    public static int GetStreamDevice(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_device(stream, out int device));
        return device;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_copy_attributes"))
    {
        methodRows.Add(
@"    public static void CopyStreamAttributes(SafeCudaStreamHandle destination, SafeCudaStreamHandle source)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_copy_attributes(destination, source));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_thread_exchange_stream_capture_mode"))
    {
        methodRows.Add(
@"    public static CudaStreamCaptureMode ExchangeThreadStreamCaptureMode(CudaStreamCaptureMode mode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_thread_exchange_stream_capture_mode((int)mode, out int previousMode));
        return (CudaStreamCaptureMode)previousMode;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_synchronize"))
    {
        methodRows.Add(
@"    public static void SynchronizeStream(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_synchronize(stream));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_query"))
    {
        methodRows.Add(
@"    public static bool QueryStream(SafeCudaStreamHandle stream)
    {
        BridgeStatusCode status = NativeMethodsCuda.jyppx_cuda_stream_query(stream);
        if (status == BridgeStatusCode.Ok)
        {
            return true;
        }

        if (status == BridgeStatusCode.NotReady)
        {
            return false;
        }

        CudaNativeStatus.ThrowIfFailed(status);
        return false;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_stream_wait_event"))
    {
        methodRows.Add(
@"    public static void WaitStreamForEvent(SafeCudaStreamHandle stream, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_wait_event(stream, eventHandle, 0));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_event_create"))
    {
        methodRows.Add(
@"    public static SafeCudaEventHandle CreateEvent()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_create(0, out SafeCudaEventHandle handle));
        return handle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_event_record"))
    {
        methodRows.Add(
@"    public static void RecordEvent(SafeCudaEventHandle eventHandle, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_record(eventHandle, stream));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_event_record_with_flags"))
    {
        methodRows.Add(
@"    public static void RecordEventWithFlags(SafeCudaEventHandle eventHandle, SafeCudaStreamHandle stream, uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_record_with_flags(eventHandle, stream, flags));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_event_query"))
    {
        methodRows.Add(
@"    public static bool QueryEvent(SafeCudaEventHandle eventHandle)
    {
        BridgeStatusCode status = NativeMethodsCuda.jyppx_cuda_event_query(eventHandle);
        if (status == BridgeStatusCode.Ok)
        {
            return true;
        }

        if (status == BridgeStatusCode.NotReady)
        {
            return false;
        }

        CudaNativeStatus.ThrowIfFailed(status);
        return false;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_event_synchronize"))
    {
        methodRows.Add(
@"    public static void SynchronizeEvent(SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_synchronize(eventHandle));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_event_elapsed_time"))
    {
        methodRows.Add(
@"    public static float GetEventElapsedTime(SafeCudaEventHandle startEvent, SafeCudaEventHandle endEvent)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_elapsed_time(startEvent, endEvent, out float milliseconds));
        return milliseconds;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_alloc"))
    {
        methodRows.Add(
@"    public static SafeCudaMemoryHandle AllocateMemory(int sizeInBytes)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_alloc((UIntPtr)sizeInBytes, out SafeCudaMemoryHandle handle));
        return handle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_alloc_managed"))
    {
        methodRows.Add(
@"    public static SafeCudaMemoryHandle AllocateManagedMemory(int sizeInBytes, uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_alloc_managed((UIntPtr)sizeInBytes, flags, out SafeCudaMemoryHandle handle));
        return handle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_get_size"))
    {
        methodRows.Add(
@"    public static ulong GetMemorySize(SafeCudaMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_get_size(memory, out UIntPtr size));
        return size.ToUInt64();
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_get_pointer_attributes"))
    {
        methodRows.Add(
@"    public static NativeCudaPointerAttributes GetPointerAttributes(SafeCudaMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_get_pointer_attributes(memory, out NativeCudaPointerAttributes attributes));
        return attributes;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_prefetch_async"))
    {
        methodRows.Add(
@"    public static void PrefetchMemoryAsync(SafeCudaMemoryHandle memory, int size, int destinationDevice, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_prefetch_async(memory, (UIntPtr)size, destinationDevice, stream));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_advise"))
    {
        methodRows.Add(
@"    public static void AdviseMemory(SafeCudaMemoryHandle memory, int size, int advice, int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_advise(memory, (UIntPtr)size, advice, device));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_memset"))
    {
        methodRows.Add(
@"    public static void FillMemory(SafeCudaMemoryHandle memory, byte value, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_memset(memory, value, (UIntPtr)count));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_memset_async"))
    {
        methodRows.Add(
@"    public static void FillMemoryAsync(SafeCudaMemoryHandle memory, byte value, int count, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_memset_async(memory, value, (UIntPtr)count, stream));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_copy_device_to_device"))
    {
        methodRows.Add(
@"    public static void CopyDeviceToDevice(SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int size)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_copy_device_to_device(destination, source, (UIntPtr)size));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_copy_device_to_device_async"))
    {
        methodRows.Add(
@"    public static void CopyDeviceToDeviceAsync(SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int size, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_copy_device_to_device_async(destination, source, (UIntPtr)size, stream));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_copy_peer"))
    {
        methodRows.Add(
@"    public static void CopyPeer(SafeCudaMemoryHandle destination, int destinationDevice, SafeCudaMemoryHandle source, int sourceDevice, int size)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_copy_peer(destination, destinationDevice, source, sourceDevice, (UIntPtr)size));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_memory_copy_peer_async"))
    {
        methodRows.Add(
@"    public static void CopyPeerAsync(SafeCudaMemoryHandle destination, int destinationDevice, SafeCudaMemoryHandle source, int sourceDevice, int size, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_copy_peer_async(destination, destinationDevice, source, sourceDevice, (UIntPtr)size, stream));
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_pinned_memory_alloc"))
    {
        methodRows.Add(
@"    public static SafeCudaPinnedMemoryHandle AllocatePinnedMemory(int sizeInBytes)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_alloc((UIntPtr)sizeInBytes, out SafeCudaPinnedMemoryHandle handle));
        return handle;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_pinned_memory_get_size"))
    {
        methodRows.Add(
@"    public static ulong GetPinnedMemorySize(SafeCudaPinnedMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_get_size(memory, out UIntPtr size));
        return size.ToUInt64();
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_pinned_memory_get_host_pointer"))
    {
        methodRows.Add(
@"    public static IntPtr GetPinnedMemoryPointer(SafeCudaPinnedMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_get_host_pointer(memory, out IntPtr pointer));
        return pointer;
    }");
    }

    if (entryPoints.Contains("jyppx_cuda_pinned_memory_get_mapped_device_pointer"))
    {
        methodRows.Add(
@"    public static IntPtr GetPinnedMemoryMappedDevicePointer(SafeCudaPinnedMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_get_mapped_device_pointer(memory, out IntPtr pointer));
        return pointer;
    }");
    }

    if (wrapperKindMap.TryGetValue("pinned-memory-copy-from-host-async", out ManifestApiRecord? copyFromPinnedAsyncApi))
    {
        methodRows.Add(
@"    public static void CopyFromPinnedHostAsync(SafeCudaMemoryHandle memory, SafeCudaPinnedMemoryHandle source, int size, SafeCudaStreamHandle stream)
    {
        IntPtr pointer = GetPinnedMemoryPointer(source);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda." + copyFromPinnedAsyncApi.EntryPoint + @"(memory, pointer, (UIntPtr)size, stream));
    }");
    }

    if (wrapperKindMap.TryGetValue("pinned-memory-copy-to-host-async", out ManifestApiRecord? copyToPinnedAsyncApi))
    {
        methodRows.Add(
@"    public static void CopyToPinnedHostAsync(SafeCudaMemoryHandle memory, SafeCudaPinnedMemoryHandle destination, int size, SafeCudaStreamHandle stream)
    {
        IntPtr pointer = GetPinnedMemoryPointer(destination);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda." + copyToPinnedAsyncApi.EntryPoint + @"(memory, pointer, (UIntPtr)size, stream));
    }");
    }

    if (wrapperKindMap.TryGetValue("pinned-byte-array-copy-from-host", out ManifestApiRecord? copyFromApi))
    {
        descriptorRows.Add(
$@"    private static readonly PinnedByteBufferDescriptor CopyFromHostDescriptor = new(
        ""CopyFromHost"",
        ""source"",
        ""{copyFromApi.EntryPoint}"",
        ""HostToDevice"");");

        methodRows.Add(
@"    public static void CopyFromHost(SafeCudaMemoryHandle memory, byte[] source, int size)
    {
        using PinnedByteBufferScope pinned = PinnedByteBufferScope.Pin(source, size, CopyFromHostDescriptor);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda." + copyFromApi.EntryPoint + @"(memory, pinned.Pointer, pinned.Size));
    }");
    }

    if (wrapperKindMap.TryGetValue("pinned-byte-array-copy-to-host", out ManifestApiRecord? copyToApi))
    {
        descriptorRows.Add(
$@"    private static readonly PinnedByteBufferDescriptor CopyToHostDescriptor = new(
        ""CopyToHost"",
        ""destination"",
        ""{copyToApi.EntryPoint}"",
        ""DeviceToHost"");");

        methodRows.Add(
@"    public static void CopyToHost(SafeCudaMemoryHandle memory, byte[] destination, int size)
    {
        using PinnedByteBufferScope pinned = PinnedByteBufferScope.Pin(destination, size, CopyToHostDescriptor);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda." + copyToApi.EntryPoint + @"(memory, pinned.Pointer, pinned.Size));
    }");
    }

    string descriptors = descriptorRows.Count == 0
        ? string.Empty
        : string.Join(Environment.NewLine + Environment.NewLine, descriptorRows) + Environment.NewLine + Environment.NewLine;
    string rows = string.Join(Environment.NewLine + Environment.NewLine, methodRows);
    if (string.IsNullOrWhiteSpace(rows))
    {
        rows = "    // No CUDA wrapper methods were generated in this pass.";
    }

    WriteAllTextAtomic(
        outputPath,
        templateText
            .Replace("{{GeneratedOn}}", generatedOn)
            .Replace("{{DescriptorRows}}", descriptors)
            .Replace("{{MethodRows}}", rows),
        new UTF8Encoding(false));
}

static void WriteTensorRtLineBindings(
    string outputPath,
    string generatedOn,
    string templateText,
    List<ManifestApiRecord> apis)
{
    string[] requiredRoles = new[]
    {
        "query-adapter-info",
        "logger-create",
        "runtime-create",
        "builder-create",
        "builder-config-create",
        "network-create",
        "serialized-build",
        "deserialize-host-memory",
        "execution-context-create"
    };

    List<string> rows = new List<string>();
    foreach (IGrouping<string, ManifestApiRecord> group in apis
                 .Where(api => api.VersionLine == "8" || api.VersionLine == "10" || api.VersionLine == "11")
                 .GroupBy(api => api.VersionLine)
                 .OrderBy(group => group.Key, StringComparer.OrdinalIgnoreCase))
    {
        Dictionary<string, ManifestApiRecord> byRole = group
            .Where(api => !string.IsNullOrWhiteSpace(api.BindingRole))
            .ToDictionary(api => api.BindingRole, api => api, StringComparer.Ordinal);

        if (requiredRoles.Any(role => !byRole.ContainsKey(role)))
        {
            continue;
        }

        string fieldName = group.Key switch
        {
            "8" => "Trt8Bindings",
            "10" => "Trt10Bindings",
            "11" => "Trt11Bindings",
            _ => throw new InvalidOperationException($"Unsupported TensorRT line for binding generation: {group.Key}")
        };
        string lineName = group.Key switch
        {
            "8" => "TensorRT 8",
            "10" => "TensorRT 10",
            "11" => "TensorRT 11",
            _ => throw new InvalidOperationException($"Unsupported TensorRT line for binding generation: {group.Key}")
        };
        string lineEnum = group.Key switch
        {
            "8" => "TensorRtApiLine.TensorRt8",
            "10" => "TensorRtApiLine.TensorRt10",
            "11" => "TensorRtApiLine.TensorRt11",
            _ => throw new InvalidOperationException($"Unsupported TensorRT line for binding generation: {group.Key}")
        };

        rows.Add(
$@"    private static readonly TensorRtLineBindings {fieldName} = new(
        {lineEnum},
        ""{lineName}"",
        NativeMethodsTensorRt.{byRole["query-adapter-info"].EntryPoint},
        NativeMethodsTensorRt.{byRole["logger-create"].EntryPoint},
        NativeMethodsTensorRt.{byRole["runtime-create"].EntryPoint},
        NativeMethodsTensorRt.{byRole["builder-create"].EntryPoint},
        NativeMethodsTensorRt.{byRole["builder-config-create"].EntryPoint},
        NativeMethodsTensorRt.{byRole["network-create"].EntryPoint},
        NativeMethodsTensorRt.{byRole["serialized-build"].EntryPoint},
        NativeMethodsTensorRt.{byRole["deserialize-host-memory"].EntryPoint},
        NativeMethodsTensorRt.{byRole["execution-context-create"].EntryPoint});");
    }

    string bindingRows = rows.Count == 0
        ? "    // No TensorRT line bindings were generated in this pass."
        : string.Join(Environment.NewLine + Environment.NewLine, rows);

    WriteAllTextAtomic(
        outputPath,
        templateText
            .Replace("{{GeneratedOn}}", generatedOn)
            .Replace("{{BindingRows}}", bindingRows),
        new UTF8Encoding(false));
}

static void WriteTensorRtLineHelpers(
    string outputPath,
    string generatedOn,
    string templateText,
    List<ManifestApiRecord> apis)
{
    bool hasQueryAdapter = apis.Any(api => api.BindingRole == "query-adapter-info");
    bool hasLoggerCreate = apis.Any(api => api.BindingRole == "logger-create");
    bool hasRuntimeCreate = apis.Any(api => api.BindingRole == "runtime-create");
    bool hasBuilderCreate = apis.Any(api => api.BindingRole == "builder-create");

    List<string> helperRows = new List<string>();

    if (hasQueryAdapter)
    {
        helperRows.Add(
@"    private static NativeTensorRtAdapterInfo GetAdapterInfoCore(TensorRtLineBindings bindings)
    {
        BridgeStatusCode status = bindings.QueryAdapterInfo(out NativeTensorRtAdapterInfo info);
        NativeStatus.ThrowIfFailed(status);
        return info;
    }");
    }

    if (hasLoggerCreate)
    {
        helperRows.Add(
@"    private static SafeTensorRtObjectHandle CreateLoggerCore(TensorRtLineBindings bindings)
    {
        BridgeStatusCode status = bindings.CreateLogger(out SafeTensorRtObjectHandle handle);
        NativeStatus.ThrowIfFailed(status);
        return handle;
    }");
    }

    if (hasRuntimeCreate)
    {
        helperRows.Add(
@"    private static bool TryCreateRuntimeCore(TensorRtLineBindings bindings, SafeTensorRtObjectHandle logger, out string message)
    {
        BridgeStatusCode status = bindings.RuntimeCreate(logger, out SafeTensorRtObjectHandle runtime);
        if (status == BridgeStatusCode.Ok)
        {
            runtime.Dispose();
            message = ""Runtime handle created successfully."";
            return true;
        }

        message = GetLastErrorMessageOrFallback($""Runtime creation failed with status '{status}'."");
        return false;
    }");
    }

    if (hasBuilderCreate)
    {
        helperRows.Add(
@"    private static bool TryCreateBuilderCore(TensorRtLineBindings bindings, out string message)
    {
        using SafeTensorRtObjectHandle logger = CreateLoggerCore(bindings);
        BridgeStatusCode status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
        if (status == BridgeStatusCode.Ok)
        {
            builder.Dispose();
            message = ""Builder handle created successfully."";
            return true;
        }

        message = GetLastErrorMessageOrFallback($""Builder creation failed with status '{status}'."");
        return false;
    }");
    }

    string rows = helperRows.Count == 0
        ? "    // No TensorRT line helper methods were generated in this pass."
        : string.Join(Environment.NewLine + Environment.NewLine, helperRows);

    WriteAllTextAtomic(
        outputPath,
        templateText
            .Replace("{{GeneratedOn}}", generatedOn)
            .Replace("{{HelperRows}}", rows),
        new UTF8Encoding(false));
}

static string SanitizeParameterName(string value)
{
    string sanitized = value
        .Replace("-", "_", StringComparison.Ordinal)
        .Replace(" ", "_", StringComparison.Ordinal);

    if (string.IsNullOrWhiteSpace(sanitized))
    {
        return "value";
    }

    if (char.IsDigit(sanitized[0]))
    {
        sanitized = "_" + sanitized;
    }

    return sanitized switch
    {
        "object" => "objectHandle",
        "event" => "eventHandle",
        "string" => "stringValue",
        "params" => "paramsValue",
        "ref" => "refValue",
        "out" => "outValue",
        "base" => "baseValue",
        "namespace" => "namespaceValue",
        "internal" => "internalValue",
        _ => sanitized
    };
}

internal sealed record ManifestApiRecord(
    string Id,
    string Module,
    string VersionLine,
    string EntryPoint,
    string ReturnType,
    string Ownership,
    bool ManualOverride,
    string VersionGuard,
    string WrapperKind,
    string BindingRole,
    IReadOnlyList<ManifestParameterRecord> Parameters);

internal sealed record ManifestParameterRecord(
    string Name,
    string NativeType,
    string Direction,
    string Ownership,
    string ManagedType,
    string ModuleManagedType);

internal sealed class ManifestDocument
{
    public string Module { get; set; } = string.Empty;
    public string VersionLine { get; set; } = string.Empty;
    public List<ManifestApi> Apis { get; set; } = new List<ManifestApi>();
}

internal sealed class ManifestApi
{
    public string Id { get; set; } = string.Empty;
    public string EntryPoint { get; set; } = string.Empty;
    public string ReturnType { get; set; } = string.Empty;
    public string Ownership { get; set; } = string.Empty;
    public bool ManualOverride { get; set; }
    public string? VersionGuard { get; set; }
    public string? WrapperKind { get; set; }
    public string? BindingRole { get; set; }
    public List<ManifestParameter> Parameters { get; set; } = new List<ManifestParameter>();
}

internal sealed class ManifestParameter
{
    public string Name { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public string Direction { get; set; } = string.Empty;
    public string? Ownership { get; set; }
    public string? ManagedType { get; set; }
    public string? ModuleManagedType { get; set; }
}
