using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static TensorRtDims ResolveRuntimeInputShape(TensorRtEngineTensorBinding input, OnnxEngineBuildOptions options)
    {
        if (options.ShapeProfile.TryGetShapeTriple(input.Name, out _, out EngineBuildShape opt, out _))
        {
            return new TensorRtDims(opt.Dimensions);
        }

        if (input.ProfileOptShape != null && CanEstimate(input.ProfileOptShape))
        {
            return input.ProfileOptShape;
        }

        if (input.EngineShape != null && CanEstimate(input.EngineShape))
        {
            return input.EngineShape;
        }

        throw new NotSupportedException($"Input '{input.Name}' does not have a concrete runtime shape. Provide --shapes or --optShapes for generic bounded runtime.");
    }

    private static bool ShouldSetInputShape(TensorRtEngineTensorBinding input, TensorRtDims runtimeShape, OnnxEngineBuildOptions options)
    {
        return !CanEstimate(input.EngineShape) ||
            input.ProfileOptShape != null ||
            !options.ShapeProfile.IsEmpty ||
            !ShapesEqual(input.EngineShape, runtimeShape);
    }

    private static float[] CreateRuntimeInputValues(
        string inputName,
        int expectedCount,
        IReadOnlyDictionary<string, string> inputs,
        int inputIndex)
    {
        if (inputs.Count == 0)
        {
            float[] generated = new float[expectedCount];
            for (int index = 0; index < generated.Length; index++)
            {
                generated[index] = checked(inputIndex * 1024 + index) + 0.5f;
            }

            return generated;
        }

        if (!inputs.TryGetValue(inputName, out string? path))
        {
            throw new ArgumentException($"--loadInputs must include a mapping for input tensor '{inputName}'.");
        }

        return ReadFloatInputData(expectedCount, path);
    }

    private static Dictionary<string, string> ParseLoadInputs(string value)
    {
        Dictionary<string, string> result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(value))
        {
            return result;
        }

        foreach (string segment in value.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int separator = segment.IndexOf(':');
            if (separator <= 0 || separator == segment.Length - 1)
            {
                throw new ArgumentException("--loadInputs entries must use tensor:path format.");
            }

            string tensorName = segment.Substring(0, separator).Trim();
            string path = Path.GetFullPath(segment.Substring(separator + 1).Trim().Trim('"'));
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Input tensor file was not found.", path);
            }

            if (!result.TryAdd(tensorName, path))
            {
                throw new ArgumentException($"--loadInputs contains a duplicate mapping for tensor '{tensorName}'.");
            }
        }

        return result;
    }

    private static float[] ReadFloatInputData(int expectedCount, string path)
    {
        string extension = Path.GetExtension(path);
        if (string.Equals(extension, ".bin", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".raw", StringComparison.OrdinalIgnoreCase))
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (bytes.Length % sizeof(float) != 0)
            {
                throw new ArgumentException($"Float input file byte length must be divisible by {sizeof(float)}.");
            }

            int actualCount = bytes.Length / sizeof(float);
            if (actualCount != expectedCount)
            {
                throw new ArgumentException($"Float input file has {actualCount} elements, expected {expectedCount}.");
            }

            float[] values = new float[actualCount];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        string text = File.ReadAllText(path);
        float[] parsed = text
            .Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(static item => float.Parse(item, System.Globalization.CultureInfo.InvariantCulture))
            .ToArray();
        if (parsed.Length != expectedCount)
        {
            throw new ArgumentException($"Text input file has {parsed.Length} float values, expected {expectedCount}.");
        }

        return parsed;
    }

}
