using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.SampleSupport;

namespace ClassificationSample;

public static class ClassificationCommand
{
    private const string DefaultInputShape = "1x3x224x224";

    public static int Run(string[] args)
    {
        if (SampleCommandLine.HasSwitch(args, "--help"))
        {
            PrintUsage();
            return 0;
        }

        try
        {
            ClassificationImagePreprocessResult? preprocess = TryPreprocessImage(args);
            string[] effectiveArgs = preprocess == null
                ? args
                : AddOrReplaceArgument(args, "--input-data", preprocess.TensorPath);
            OnnxSampleOptions options = OnnxSampleOptions.FromArgs(effectiveArgs, DefaultInputShape);
            string labelsPath = ResolveOptionalFile(SampleCommandLine.GetStringArgument(args, "--labels", string.Empty), "Labels file");
            IReadOnlyList<string> labels = TensorRtOnnxSample.ReadLabels(labelsPath);
            int topK = Math.Max(1, SampleCommandLine.GetPositiveIntArgument(args, "--top-k", 5));
            ClassificationScoreTransform transform = ClassificationOutputProcessor.ParseScoreTransform(
                SampleCommandLine.GetStringArgument(args, "--score-transform", "raw"));
            OnnxSampleResult result = TensorRtOnnxSample.RunSingleFloatInputOutput(options);
            float[] transformedValues = ClassificationOutputProcessor.Transform(result.OutputValues, transform);
            IReadOnlyList<ClassificationPrediction> predictions = ClassificationOutputProcessor.GetTopK(transformedValues, labels, topK);

            string modelSha256 = ClassificationOutputProcessor.ComputeFileSha256OrEmpty(options.ModelPath);
            string labelsSha256 = ClassificationOutputProcessor.ComputeFileSha256OrEmpty(labelsPath);
            string inputTensorSha256 = ResolveInputTensorSha256(options, preprocess);
            string preprocessContractSha256 = preprocess?.Options.ContractSha256 ??
                ParseOptionalSha256(args, "--preprocess-contract-sha256");
            string valueKind = ClassificationOutputProcessor.ValueKind(transform);
            int[] outputShape = result.OutputShape.Values;
            string outputTensorContractSha256 = ClassificationOutputProcessor.ComputeOutputTensorContractSha256(
                result.OutputName,
                outputShape,
                valueKind);
            string taskSemanticsSha256 = ClassificationOutputProcessor.ComputeTaskSemanticsSha256(transform, topK, labelsSha256);
            ClassificationReferenceContext referenceContext = new ClassificationReferenceContext(
                result.OutputName,
                outputShape,
                valueKind,
                modelSha256,
                inputTensorSha256,
                preprocessContractSha256,
                outputTensorContractSha256,
                labelsSha256,
                taskSemanticsSha256);
            string referencePath = SampleCommandLine.GetStringArgument(args, "--reference-output", string.Empty);
            ClassificationReferenceValidationResult validation = ClassificationOutputProcessor.ValidateReference(
                referencePath,
                referenceContext,
                transformedValues,
                ParseNonNegativeFloat(args, "--reference-abs", 0.0f),
                ParseNonNegativeFloat(args, "--reference-rel", 0.0f),
                SampleCommandLine.GetStringArgument(args, "--reference-nan-policy", "reject"),
                SampleCommandLine.GetStringArgument(args, "--reference-infinity-policy", "exact"));

            string outputJsonPath = SampleCommandLine.GetStringArgument(args, "--output-json", string.Empty);
            ClassificationOutputReportWriter.Write(
                outputJsonPath,
                options,
                result,
                preprocess,
                labelsPath,
                labelsSha256,
                labels.Count,
                transform,
                transformedValues,
                predictions,
                referenceContext,
                validation);

            string visualizationPath = SampleCommandLine.GetStringArgument(
                args,
                "--visualization",
                SampleCommandLine.GetStringArgument(args, "--visualization-svg", string.Empty));
            string visualizationBackgroundPath = SampleCommandLine.GetStringArgument(
                args,
                "--visualization-background",
                string.Empty);
            if (!string.IsNullOrWhiteSpace(visualizationBackgroundPath) && string.IsNullOrWhiteSpace(visualizationPath))
            {
                throw new ArgumentException("--visualization-background requires --visualization <path>.");
            }
            if (!string.IsNullOrWhiteSpace(visualizationPath))
            {
                if (preprocess == null)
                {
                    throw new ArgumentException("--visualization requires --image so source image dimensions are available.");
                }
                if (string.IsNullOrWhiteSpace(visualizationBackgroundPath))
                {
                    visualizationBackgroundPath = preprocess.SourcePath;
                }
                ClassificationVisualizationWriter.Write(
                    visualizationPath,
                    ResolveOptionalFile(visualizationBackgroundPath, "Classification visualization background"),
                    preprocess,
                    predictions);
            }

            Console.WriteLine($"Classification TensorRtLine={(int)result.Line} Model={options.ModelPath}");
            Console.WriteLine($"Input={result.InputName}:{result.InputShape} Inputs={result.Inputs.Count} Output={result.OutputName}:{result.OutputShape}");
            foreach (OnnxSampleInputTensor input in result.Inputs)
            {
                Console.WriteLine($"RuntimeInput Tensor={input.Name} Shape={input.Shape} Elements={input.ElementCount} Bytes={input.ByteLength} Source={input.SourceClassification} SourcePath={input.SourcePath} Sha256={input.Sha256}");
            }
            Console.WriteLine($"ProfileIndex={result.ProfileIndex} EngineDeviceMemory={result.EngineDeviceMemoryBytes}");
            Console.WriteLine($"Execution {result.ExecutionSummary} ElapsedMs={result.ElapsedMilliseconds:0.###}");
            Console.WriteLine($"InputSource={(preprocess != null ? "image" : options.UsesExternalInput ? "external-tensor" : "synthetic")} TensorSha256={inputTensorSha256} PreprocessContractSha256={preprocessContractSha256}");
            if (preprocess != null)
            {
                PrintPreprocess(preprocess);
            }
            Console.WriteLine($"ScoreTransform={transform.ToString().ToLowerInvariant()} ValueKind={valueKind} OutputSha256={ClassificationOutputProcessor.ComputeFloatSha256(transformedValues)}");
            foreach (ClassificationPrediction prediction in predictions)
            {
                Console.WriteLine($"TopK Index={prediction.Index} Label={prediction.Label} Score={prediction.Score:0.######}");
            }
            if (validation.Requested)
            {
                Console.WriteLine($"ClassificationReference Requested=True Completed={validation.Completed} Passed={validation.Passed} Compared={validation.ComparedElementCount} Mismatches={validation.MismatchCount} FirstMismatch={validation.FirstMismatchIndex} Diagnostic={validation.Diagnostic}");
            }
            if (result.ReferenceValidation.Requested)
            {
                Console.WriteLine($"ReferenceOutputValidation Requested=True Completed={result.ReferenceValidation.Completed} Passed={result.ReferenceValidation.Passed} Tensors={result.ReferenceValidation.TensorComparisons.Count} AbsTolerance={result.ReferenceValidation.AbsoluteTolerance:R} RelTolerance={result.ReferenceValidation.RelativeTolerance:R} NaNPolicy={result.ReferenceValidation.NaNPolicy} InfinityPolicy={result.ReferenceValidation.InfinityPolicy}");
            }
            bool validationRequested = validation.Requested || result.ReferenceValidation.Requested;
            bool validationPassed = (!validation.Requested || validation.Passed) &&
                (!result.ReferenceValidation.Requested || (result.ReferenceValidation.Completed && result.ReferenceValidation.Passed));
            Console.WriteLine("OutputValidated=" + (validationRequested && validationPassed));
            if (!string.IsNullOrWhiteSpace(outputJsonPath))
            {
                Console.WriteLine("OutputJson=" + Path.GetFullPath(outputJsonPath));
            }
            if (!string.IsNullOrWhiteSpace(visualizationPath))
            {
                Console.WriteLine("Visualization=" + Path.GetFullPath(visualizationPath));
            }

            bool success = validationPassed;
            Console.WriteLine("Classification Passed=" + success);
            return success ? 0 : 1;
        }
        catch (SampleSkippedException exception)
        {
            Console.WriteLine($"Classification=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (Exception exception) when (TensorRtSampleSupport.IsDeploymentException(exception))
        {
            Console.WriteLine($"Classification=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (Exception exception) when (exception is ArgumentException || exception is FileNotFoundException || exception is InvalidDataException || exception is JsonException)
        {
            Console.WriteLine($"Classification=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
    }

    private static ClassificationImagePreprocessResult? TryPreprocessImage(string[] args)
    {
        string imagePath = SampleCommandLine.GetStringArgument(args, "--image", string.Empty);
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return null;
        }
        if (!string.IsNullOrWhiteSpace(SampleCommandLine.GetStringArgument(args, "--input", string.Empty)) ||
            !string.IsNullOrWhiteSpace(SampleCommandLine.GetStringArgument(args, "--input-data", string.Empty)))
        {
            throw new ArgumentException("--image cannot be combined with --input or --input-data.");
        }

        string fullImagePath = ResolveOptionalFile(imagePath, "Classification image");
        string tensorPath = SampleCommandLine.GetStringArgument(args, "--preprocessed-output", string.Empty);
        if (string.IsNullOrWhiteSpace(tensorPath))
        {
            tensorPath = Path.Combine(
                Path.GetDirectoryName(fullImagePath) ?? string.Empty,
                Path.GetFileNameWithoutExtension(fullImagePath) + ".classification-f32.bin");
        }

        int[] inputShape = TensorRtOnnxSample.ParseShape(
            SampleCommandLine.GetStringArgument(args, "--input-shape", DefaultInputShape),
            "--input-shape").Values;
        ClassificationPreprocessOptions options = new ClassificationPreprocessOptions(
            SampleCommandLine.GetStringArgument(args, "--image-resize", "shorter-side-center-crop"),
            SampleCommandLine.GetPositiveIntArgument(args, "--resize-shorter-side", 256),
            SampleCommandLine.GetStringArgument(args, "--tensor-layout", "NCHW"),
            SampleCommandLine.GetStringArgument(args, "--color-order", "RGB"),
            ParsePositiveFloat(args, "--scale", 1.0f / 255.0f),
            ParseFloatTriplet(args, "--mean", new[] { 0.485f, 0.456f, 0.406f }),
            ParseFloatTriplet(args, "--std", new[] { 0.229f, 0.224f, 0.225f }));
        return ClassificationImagePreprocessor.Preprocess(fullImagePath, tensorPath, inputShape, options);
    }

    private static string ResolveInputTensorSha256(OnnxSampleOptions options, ClassificationImagePreprocessResult? preprocess)
    {
        if (preprocess != null)
        {
            return preprocess.TensorSha256;
        }
        float[] inputValues = TensorRtOnnxSample.CreateInputValuesForTesting(
            TensorRtOnnxSample.CountElements(options.InputShape),
            options.InputPattern,
            options.InputPath,
            options.InputDataPath);
        return ClassificationOutputProcessor.ComputeFloatSha256(inputValues);
    }

    private static void PrintPreprocess(ClassificationImagePreprocessResult result)
    {
        Console.WriteLine($"ImagePreprocess Source={result.SourcePath} SourceSha256={result.SourceSha256} SourceSize={result.SourceWidth}x{result.SourceHeight} Tensor={result.TensorPath} TensorSha256={result.TensorSha256}");
        Console.WriteLine($"ImagePreprocessConfig Mode={result.Options.ResizeMode} ShorterSide={result.Options.ResizeShorterSide} Resized={result.ResizedWidth}x{result.ResizedHeight} Crop={result.CropX},{result.CropY},{result.TargetWidth},{result.TargetHeight} Layout={result.Options.TensorLayout} Color={result.Options.ColorOrder} Scale={result.Options.Scale:R} Mean={FormatTriplet(result.Options.Mean)} Std={FormatTriplet(result.Options.StandardDeviation)} ContractSha256={result.Options.ContractSha256}");
    }

    private static string[] AddOrReplaceArgument(string[] args, string name, string value)
    {
        var result = new List<string>();
        for (int index = 0; index < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 < args.Length && !args[index + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    index++;
                }
                continue;
            }
            result.Add(args[index]);
        }
        result.Add(name);
        result.Add(value);
        return result.ToArray();
    }

    private static string ResolveOptionalFile(string path, string description)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }
        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException(description + " was not found.", fullPath);
        }
        return fullPath;
    }

    private static float ParsePositiveFloat(string[] args, string name, float defaultValue)
    {
        float value = ParseFloat(args, name, defaultValue);
        if (value <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(name, "Value must be positive.");
        }
        return value;
    }

    private static float ParseNonNegativeFloat(string[] args, string name, float defaultValue)
    {
        float value = ParseFloat(args, name, defaultValue);
        if (value < 0.0f)
        {
            throw new ArgumentOutOfRangeException(name, "Value must be non-negative.");
        }
        return value;
    }

    private static float ParseFloat(string[] args, string name, float defaultValue)
    {
        string text = SampleCommandLine.GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(text))
        {
            return defaultValue;
        }
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) || !float.IsFinite(value))
        {
            throw new ArgumentException(name + " must be a finite floating-point value.");
        }
        return value;
    }

    private static float[] ParseFloatTriplet(string[] args, string name, float[] defaultValue)
    {
        string text = SampleCommandLine.GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(text))
        {
            return (float[])defaultValue.Clone();
        }
        string[] tokens = text.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length != 3)
        {
            throw new ArgumentException(name + " must contain three comma-separated values.");
        }
        return tokens.Select(token =>
        {
            if (!float.TryParse(token.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float value) || !float.IsFinite(value))
            {
                throw new ArgumentException(name + " must contain finite floating-point values.");
            }
            return value;
        }).ToArray();
    }

    private static string ParseOptionalSha256(string[] args, string name)
    {
        string value = SampleCommandLine.GetStringArgument(args, name, string.Empty).Trim().ToLowerInvariant();
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }
        if (value.Length != 64 || value.Any(static character =>
            (character < '0' || character > '9') && (character < 'a' || character > 'f')))
        {
            throw new ArgumentException(name + " must be a 64-character SHA256 value.");
        }

        return value;
    }

    private static string FormatTriplet(float[] values)
    {
        return string.Join(",", values.Select(static value => value.ToString("R", CultureInfo.InvariantCulture)));
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Classification sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/Classification -- --model model.onnx --labels labels.txt --image image.ppm --preprocessed-output input-f32.bin --output-json output.json --input-shape 1x3x224x224 --tensor-rt-line 10");
        Console.WriteLine("Input options:");
        Console.WriteLine("  --image <bmp|ppm>         Decode and preprocess a real image; cannot be combined with --input/--input-data.");
        Console.WriteLine("  --preprocessed-output <path>  Float32 tensor written for --image. Defaults beside the image.");
        Console.WriteLine("  --image-resize <mode>     shorter-side-center-crop or stretch.");
        Console.WriteLine("  --resize-shorter-side <n> Default: 256. Used by center-crop mode.");
        Console.WriteLine("  --tensor-layout <layout>  NCHW or NHWC. Default: NCHW.");
        Console.WriteLine("  --color-order <order>     RGB or BGR. Default: RGB.");
        Console.WriteLine("  --scale <value>           Applied before mean/std. Default: 1/255.");
        Console.WriteLine("  --mean <r,g,b> --std <r,g,b>  Default ImageNet normalization.");
        Console.WriteLine("  --input-pattern <pattern> zeros, ones, or ramp. Synthetic pipeline evidence only.");
        Console.WriteLine("  --input <path>            Raw byte tensor normalized to [0,1]; byte count must match input elements.");
        Console.WriteLine("  --input-data <path>       Preprocessed float32 .bin/.raw or text tensor.");
        Console.WriteLine("Runtime/output options:");
        Console.WriteLine("  --input-name/--output-name <name> --min-shape/--opt-shape/--max-shape <dims>");
        Console.WriteLine("  --input-shapes <map> with --load-inputs/--load-byte-inputs/--input-patterns <map> enables strict named multi-input binding.");
        Console.WriteLine("  --min-shapes/--opt-shapes/--max-shapes <map> provide the complete named dynamic profile.");
        Console.WriteLine("  --score-transform <raw|softmax> --top-k <n> --output-json <path>");
        Console.WriteLine("  --visualization <path> Write an SVG with Top-K predictions over the source image.");
        Console.WriteLine("  --visualization-svg <path> Alias for --visualization.");
        Console.WriteLine("  --visualization-background <jpg|png|bmp> Same-size background; required when --image uses PPM.");
        Console.WriteLine("Reference options:");
        Console.WriteLine("  --reference-output <json> --reference-abs <n> --reference-rel <n>");
        Console.WriteLine("  --reference-nan-policy <reject|equal> --reference-infinity-policy <exact|reject>");
        Console.WriteLine("  --reference-outputs <tensor:path,...> validates every raw runtime output; use --reference-abs-tolerance/--reference-rel-tolerance.");
        Console.WriteLine("  --preprocess-contract-sha256 <hash> is required for reference validation of externally preprocessed tensors.");
    }
}
