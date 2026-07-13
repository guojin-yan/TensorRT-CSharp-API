using System.IO;
using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxConfigScalarControlsTests
{
    [Fact]
    public void OnnxConfigScalarManifestsPromoteRealAbiWithoutDeletingDeferredHistory()
    {
        string manifest8 = ReadTensorRtManifest("v8", "trt8-onnx-config-safe-scalar-controls.manifest.json");
        string manifest10 = ReadTensorRtManifest("v10", "trt10-onnx-config-safe-scalar-controls.manifest.json");
        string manifest11 = ReadTensorRtManifest("v11", "trt11-onnx-config-safe-scalar-controls.manifest.json");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11 })
        {
            Assert.Contains("onnx-config-create", manifest);
            Assert.Contains("onnx-config-get-model-dtype", manifest);
            Assert.Contains("onnx-config-set-model-dtype", manifest);
            Assert.Contains("onnx-config-get-verbosity-level", manifest);
            Assert.Contains("onnx-config-set-verbosity-level", manifest);
            Assert.Contains("onnx-config-add-verbosity", manifest);
            Assert.Contains("onnx-config-reduce-verbosity", manifest);
            Assert.Contains("onnx-config-get-model-file-name", manifest);
            Assert.Contains("onnx-config-set-model-file-name", manifest);
            Assert.Contains("onnx-config-get-text-file-name", manifest);
            Assert.Contains("onnx-config-set-text-file-name", manifest);
            Assert.Contains("onnx-config-get-full-text-file-name", manifest);
            Assert.Contains("onnx-config-set-full-text-file-name", manifest);
            Assert.Contains("onnx-config-get-print-layer-info", manifest);
            Assert.Contains("onnx-config-set-print-layer-info", manifest);
            Assert.Contains("\"type\": \"JYPPX_TensorRtOnnxConfig*\"", manifest);
            Assert.Contains("\"type\": \"JYPPX_Boolean*\", \"direction\": \"out\", \"managedType\": \"out int\"", manifest);
            Assert.Contains("\"type\": \"char*\", \"direction\": \"out\", \"managedType\": \"byte[]\"", manifest);
            Assert.Contains("\"type\": \"const char*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", manifest);
            Assert.Contains("\"type\": \"size_t*\", \"direction\": \"out\", \"managedType\": \"out UIntPtr\"", manifest);
            Assert.DoesNotContain("_deferred", manifest);
        }

        Assert.Contains("\"versionGuard\": \"JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8\"", manifest8);
        Assert.Contains("\"versionGuard\": \"JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10\"", manifest10);
        Assert.Contains("\"versionGuard\": \"JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11\"", manifest11);

        string deferred8 = ReadTensorRtManifest("v8", "trt8-cross-version-eighth-batch-onnx-config-parser-deferred.manifest.json");
        string deferred10Global = ReadTensorRtManifest("v10", "trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json");
        string deferred10Coverage = ReadTensorRtManifest("v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");
        string deferred11Global = ReadTensorRtManifest("v11", "trt11-forty-fourth-batch-global-deferred.manifest.json");
        string deferred11Coverage = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("trt8-onnx-config-get-model-dtype-deferred", deferred8);
        Assert.Contains("trt8-onnx-config-set-model-dtype-deferred", deferred8);
        Assert.Contains("trt8-onnx-config-get-print-layer-info-deferred", deferred8);
        Assert.Contains("trt8-onnx-config-set-print-layer-info-deferred", deferred8);
        Assert.Contains("trt8-onnx-config-get-model-file-name-deferred", deferred8);
        Assert.Contains("trt8-onnx-config-set-full-text-file-name-deferred", deferred8);
        Assert.Contains("trt10-onnx-config-create-deferred", deferred10Global);
        Assert.Contains("trt10-onnx-config-get-verbosity-level-deferred", deferred10Global);
        Assert.Contains("trt10-onnx-config-set-verbosity-level-deferred", deferred10Global);
        Assert.Contains("trt10-onnx-config-reduce-verbosity-deferred", deferred10Global);
        Assert.Contains("trt10-onnx-config-get-model-dtype-deferred", deferred10Coverage);
        Assert.Contains("trt10-onnx-config-get-print-layer-info-deferred", deferred10Coverage);
        Assert.Contains("trt11-onnx-config-create-deferred", deferred11Global);
        Assert.Contains("trt11-onnx-config-get-model-dtype-deferred", deferred11Coverage);
        Assert.Contains("trt11-onnx-config-get-model-file-name-deferred", deferred11Coverage);
        Assert.Contains("trt11-onnx-config-set-text-file-name-deferred", deferred11Coverage);
        Assert.Contains("trt11-onnx-config-set-print-layer-info-deferred", deferred11Coverage);
    }

    [Fact]
    public void NativeOnnxConfigImplementationUsesOwnedHandleAndSafeScalarAccessors()
    {
        string types = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string objectSource = ReadSource("native", "src", "tensorrt", "common", "object.cpp");
        string controls = ReadSource("native", "src", "tensorrt", "common", "onnx_config_controls.inc");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string api8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string api10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string api11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_ONNX_CONFIG", types);
        Assert.Contains("typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOnnxConfig;", types);
        Assert.Contains("onnx-config", objectSource);

        Assert.Contains("BridgeOwnedOnnxConfig final : public nvonnxparser::IOnnxConfig", controls);
        Assert.Contains("new (std::nothrow) BridgeOwnedOnnxConfig()", controls);
        Assert.Contains("config->getModelDtype()", controls);
        Assert.Contains("config->setModelDtype(static_cast<nvinfer1::DataType>(data_type))", controls);
        Assert.Contains("config->getVerbosityLevel()", controls);
        Assert.Contains("config->setVerbosityLevel(static_cast<nvonnxparser::IOnnxConfig::Verbosity>(verbosity))", controls);
        Assert.Contains("config->addVerbosity()", controls);
        Assert.Contains("config->reduceVerbosity()", controls);
        Assert.Contains("config->getModelFileName()", controls);
        Assert.Contains("config->setModelFileName(value)", controls);
        Assert.Contains("config->getTextFileName()", controls);
        Assert.Contains("config->setTextFileName(value)", controls);
        Assert.Contains("config->getFullTextFileName()", controls);
        Assert.Contains("config->setFullTextFileName(value)", controls);
        Assert.Contains("config->getPrintLayerInfo()", controls);
        Assert.Contains("config->setPrintLayerInfo(enabled != JYPPX_FALSE)", controls);
        Assert.Contains("copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size)", controls);
        Assert.Contains("ONNX config model dtype must be Float, Half, or Int8.", controls);
        Assert.Contains("ONNX config verbosity must be greater than or equal to zero.", controls);
        Assert.Contains("create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ONNX_CONFIG, config, &destroy_onnx_config_payload)", controls);
        Assert.Contains("char const* getModelFileName() const noexcept override", controls);
        Assert.Contains("char const* getTextFileName() const noexcept override", controls);
        Assert.Contains("char const* getFullTextFileName() const noexcept override", controls);

        foreach (string header in new[] { header8, header10, header11 })
        {
            Assert.Contains("onnx_config_create(JYPPX_TensorRtOnnxConfig** out_config)", header);
            Assert.Contains("onnx_config_get_model_dtype(JYPPX_TensorRtOnnxConfig* config, int32_t* out_data_type)", header);
            Assert.Contains("onnx_config_set_model_dtype(JYPPX_TensorRtOnnxConfig* config, int32_t data_type)", header);
            Assert.Contains("onnx_config_get_verbosity_level(JYPPX_TensorRtOnnxConfig* config, int32_t* out_verbosity)", header);
            Assert.Contains("onnx_config_set_verbosity_level(JYPPX_TensorRtOnnxConfig* config, int32_t verbosity)", header);
            Assert.Contains("onnx_config_add_verbosity(JYPPX_TensorRtOnnxConfig* config)", header);
            Assert.Contains("onnx_config_reduce_verbosity(JYPPX_TensorRtOnnxConfig* config)", header);
            Assert.Contains("onnx_config_get_model_file_name(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)", header);
            Assert.Contains("onnx_config_set_model_file_name(JYPPX_TensorRtOnnxConfig* config, const char* value)", header);
            Assert.Contains("onnx_config_get_text_file_name(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)", header);
            Assert.Contains("onnx_config_set_text_file_name(JYPPX_TensorRtOnnxConfig* config, const char* value)", header);
            Assert.Contains("onnx_config_get_full_text_file_name(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)", header);
            Assert.Contains("onnx_config_set_full_text_file_name(JYPPX_TensorRtOnnxConfig* config, const char* value)", header);
            Assert.Contains("onnx_config_get_print_layer_info(JYPPX_TensorRtOnnxConfig* config, JYPPX_Boolean* out_enabled)", header);
            Assert.Contains("onnx_config_set_print_layer_info(JYPPX_TensorRtOnnxConfig* config, JYPPX_Boolean enabled)", header);
        }

        foreach (string api in new[] { api8, api10, api11 })
        {
            Assert.Contains("#include <NvOnnxConfig.h>", api);
            Assert.Contains("destroy_onnx_config_payload", api);
            Assert.Contains("onnx_config_controls.inc", api);
        }

        Assert.Contains("static_cast<nvonnxparser::IOnnxConfig*>(payload)->destroy();", api8);
        Assert.Contains("delete static_cast<nvonnxparser::IOnnxConfig*>(payload);", api10);
        Assert.Contains("delete static_cast<nvonnxparser::IOnnxConfig*>(payload);", api11);
    }

    [Fact]
    public void ManagedOnnxConfigWrapperRoutesAllTensorRtLinesWithoutExposingBorrowedPointers()
    {
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OnnxConfig.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxConfig.cs");
        string generated = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");

        foreach (string version in new[] { "trt8", "trt10", "trt11" })
        {
            Assert.Contains($"jyppx_{version}_onnx_config_create(out SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_get_model_dtype(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_set_model_dtype(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_get_verbosity_level(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_set_verbosity_level(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_add_verbosity(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_reduce_verbosity(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_get_model_file_name(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_set_model_file_name(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_get_text_file_name(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_set_text_file_name(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_get_full_text_file_name(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_set_full_text_file_name(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_get_print_layer_info(SafeTensorRtObjectHandle", generated);
            Assert.Contains($"jyppx_{version}_onnx_config_set_print_layer_info(SafeTensorRtObjectHandle", generated);

            Assert.Contains($"NativeMethodsTensorRt.jyppx_{version}_onnx_config_create", interop);
            Assert.Contains($"NativeMethodsTensorRt.jyppx_{version}_onnx_config_get_model_dtype", interop);
            Assert.Contains($"NativeMethodsTensorRt.jyppx_{version}_onnx_config_add_verbosity", interop);
            Assert.Contains($"NativeMethodsTensorRt.jyppx_{version}_onnx_config_get_model_file_name", interop);
            Assert.Contains($"NativeMethodsTensorRt.jyppx_{version}_onnx_config_set_full_text_file_name", interop);
            Assert.Contains($"NativeMethodsTensorRt.jyppx_{version}_onnx_config_set_print_layer_info", interop);
        }

        Assert.Contains("public TensorRtDataType ModelDataType", wrapper);
        Assert.Contains("public int VerbosityLevel", wrapper);
        Assert.Contains("public string ModelFileName", wrapper);
        Assert.Contains("public string TextFileName", wrapper);
        Assert.Contains("public string FullTextFileName", wrapper);
        Assert.Contains("public bool PrintLayerInfo", wrapper);
        Assert.Contains("public void IncreaseVerbosity()", wrapper);
        Assert.Contains("public void DecreaseVerbosity()", wrapper);
        Assert.Contains("ReadOnnxConfigString", interop);
        Assert.Contains("Utf8Interop.ToNativeString(value ?? string.Empty)", interop);
        Assert.Contains("ONNX config model data type must be Float, Half, or Int8.", interop);
        Assert.Contains("ONNX config verbosity must be greater than or equal to zero.", interop);
        Assert.Contains("SafeTensorRtObjectHandle", wrapper);
        Assert.DoesNotContain("public IntPtr", wrapper);
        Assert.DoesNotContain("public nint", wrapper);
    }

    [Fact]
    public void OnnxConfigSmokeRunnerIsRegisteredAndExercisesManagedScalarPath()
    {
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");
        string program = ReadSource("smoke", "OnnxConfigSmokeRunner", "Program.cs");

        Assert.Contains("OnnxConfigSmokeRunner", smokeReadme);
        Assert.Contains("OnnxConfigSmokeRunner.csproj", solution);
        Assert.Contains("new TensorRtOnnxConfig(line)", program);
        Assert.Contains("config.ModelDataType = TensorRtDataType.Float;", program);
        Assert.Contains("config.ModelDataType = TensorRtDataType.Half;", program);
        Assert.Contains("config.ModelDataType = TensorRtDataType.Int8;", program);
        Assert.Contains("config.VerbosityLevel", program);
        Assert.Contains("config.IncreaseVerbosity();", program);
        Assert.Contains("config.DecreaseVerbosity();", program);
        Assert.Contains("config.ModelFileName = \"models/yolovision.onnx\";", program);
        Assert.Contains("config.TextFileName = \"artifacts/onnx-parser.txt\";", program);
        Assert.Contains("config.FullTextFileName = \"artifacts/onnx-parser-full.txt\";", program);
        Assert.Contains("OnnxConfig FileNames", program);
        Assert.Contains("config.PrintLayerInfo", program);
        Assert.Contains("ManagedValidation=ModelDataType", program);
        Assert.Contains("ManagedValidation=Verbosity", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
