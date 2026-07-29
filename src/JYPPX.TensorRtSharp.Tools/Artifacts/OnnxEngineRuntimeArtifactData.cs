using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineRuntimeArtifactData
{
    public OnnxEngineRuntimeArtifactData(
        string tensorName,
        IReadOnlyList<int> shape,
        int inputElementCount,
        int outputElementCount,
        IReadOnlyList<float> inputPreview,
        IReadOnlyList<float> outputPreview,
        string executionSummary,
        byte[] rawOutputBytes,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        TensorName = tensorName ?? string.Empty;
        Shape = shape ?? Array.Empty<int>();
        InputElementCount = inputElementCount;
        OutputElementCount = outputElementCount;
        InputPreview = inputPreview ?? Array.Empty<float>();
        OutputPreview = outputPreview ?? Array.Empty<float>();
        ExecutionSummary = executionSummary ?? string.Empty;
        RawOutputBytes = rawOutputBytes ?? Array.Empty<byte>();
        TimingSamplesMilliseconds = timingSamplesMilliseconds ?? Array.Empty<float>();
        OutputTensors = CreateLegacyOutputTensors(
            TensorName,
            Shape,
            OutputElementCount,
            OutputPreview,
            RawOutputBytes);
    }

    private OnnxEngineRuntimeArtifactData(
        IReadOnlyList<OnnxEngineRuntimeInputArtifact> inputTensors,
        IReadOnlyList<OnnxEngineRuntimeOutputArtifact> outputTensors,
        OnnxEngineReferenceValidationArtifact referenceValidation,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds)
    {
        InputTensors = inputTensors?.ToArray() ?? Array.Empty<OnnxEngineRuntimeInputArtifact>();
        InputElementCount = InputTensors.Sum(static input => input.ElementCount);
        InputPreview = InputTensors.FirstOrDefault()?.Preview ?? Array.Empty<float>();
        OutputTensors = outputTensors?.ToArray() ?? Array.Empty<OnnxEngineRuntimeOutputArtifact>();
        ReferenceValidation = referenceValidation ?? OnnxEngineReferenceValidationArtifact.NotRequested;
        ExecutionSummary = executionSummary ?? string.Empty;
        TimingSamplesMilliseconds = timingSamplesMilliseconds ?? Array.Empty<float>();

        OnnxEngineRuntimeOutputArtifact? primary = OutputTensors.FirstOrDefault();
        TensorName = primary?.TensorName ?? string.Empty;
        Shape = primary?.Shape ?? Array.Empty<int>();
        OutputElementCount = primary?.ElementCount ?? 0;
        OutputPreview = primary?.Preview ?? Array.Empty<float>();
        RawOutputBytes = CombineRawOutputs(OutputTensors);
    }

    public static OnnxEngineRuntimeArtifactData Empty { get; } = new OnnxEngineRuntimeArtifactData(
        string.Empty,
        Array.Empty<int>(),
        0,
        0,
        Array.Empty<float>(),
        Array.Empty<float>(),
        string.Empty,
        Array.Empty<byte>(),
        Array.Empty<float>());

    public static OnnxEngineRuntimeArtifactData CreateIdentityOutput(
        string tensorName,
        IReadOnlyList<int> shape,
        IReadOnlyList<float> inputValues,
        IReadOnlyList<float> outputValues,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            tensorName,
            shape,
            inputValues?.Count ?? 0,
            outputValues?.Count ?? 0,
            Preview(inputValues),
            Preview(outputValues),
            executionSummary,
            ToBytes(outputValues),
            timingSamplesMilliseconds ?? Array.Empty<float>());
    }

    public static OnnxEngineRuntimeArtifactData CreateOutputSummary(
        string tensorName,
        IReadOnlyList<int> shape,
        int inputElementCount,
        IReadOnlyList<float> outputValues,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            tensorName,
            shape,
            inputElementCount,
            outputValues?.Count ?? 0,
            Array.Empty<float>(),
            Preview(outputValues),
            executionSummary,
            ToBytes(outputValues),
            timingSamplesMilliseconds ?? Array.Empty<float>());
    }

    public static OnnxEngineRuntimeArtifactData CreateOutputSummaries(
        int inputElementCount,
        IReadOnlyList<float> inputValues,
        IReadOnlyList<OnnxEngineRuntimeOutputArtifact> outputTensors,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            inputElementCount <= 0
                ? Array.Empty<OnnxEngineRuntimeInputArtifact>()
                : new[]
                {
                    new OnnxEngineRuntimeInputArtifact(
                        string.Empty,
                        Array.Empty<int>(),
                        inputElementCount,
                        Preview(inputValues),
                        ToBytes(inputValues),
                        "legacy-unspecified",
                        string.Empty)
                },
            outputTensors,
            OnnxEngineReferenceValidationArtifact.NotRequested,
            executionSummary,
            timingSamplesMilliseconds);
    }

    public static OnnxEngineRuntimeArtifactData CreateRuntimeEvidence(
        IReadOnlyList<OnnxEngineRuntimeInputArtifact> inputTensors,
        IReadOnlyList<OnnxEngineRuntimeOutputArtifact> outputTensors,
        OnnxEngineReferenceValidationArtifact referenceValidation,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            inputTensors,
            outputTensors,
            referenceValidation,
            executionSummary,
            timingSamplesMilliseconds);
    }

    internal static OnnxEngineRuntimeArtifactData CreateBenchmarkOnly(
        int inputElementCount,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            string.Empty,
            Array.Empty<int>(),
            inputElementCount,
            0,
            Array.Empty<float>(),
            Array.Empty<float>(),
            executionSummary,
            Array.Empty<byte>(),
            timingSamplesMilliseconds ?? Array.Empty<float>());
    }

    internal static OnnxEngineRuntimeArtifactData CreateBenchmarkOnly(
        IReadOnlyList<OnnxEngineRuntimeInputArtifact> inputTensors,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            inputTensors,
            Array.Empty<OnnxEngineRuntimeOutputArtifact>(),
            OnnxEngineReferenceValidationArtifact.NotRequested,
            executionSummary,
            timingSamplesMilliseconds);
    }

    public string TensorName { get; }

    public IReadOnlyList<int> Shape { get; }

    public int InputElementCount { get; }

    public int OutputElementCount { get; }

    public IReadOnlyList<float> InputPreview { get; }

    public IReadOnlyList<float> OutputPreview { get; }

    public string ExecutionSummary { get; }

    public byte[] RawOutputBytes { get; }

    public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

    public IReadOnlyList<OnnxEngineRuntimeInputArtifact> InputTensors { get; } = Array.Empty<OnnxEngineRuntimeInputArtifact>();

    public IReadOnlyList<OnnxEngineRuntimeOutputArtifact> OutputTensors { get; }

    public OnnxEngineReferenceValidationArtifact ReferenceValidation { get; } = OnnxEngineReferenceValidationArtifact.NotRequested;

    public bool OutputValidated => ReferenceValidation.Requested && ReferenceValidation.Completed && ReferenceValidation.Passed;

    public bool HasOutput => OutputTensors.Any(static output => output.ElementCount > 0 && output.Preview.Count > 0);

    public bool HasRawOutput => RawOutputBytes.Length > 0;

    private static IReadOnlyList<float> Preview(IReadOnlyList<float>? values)
    {
        if (values == null || values.Count == 0)
        {
            return Array.Empty<float>();
        }

        return values.Take(8).ToArray();
    }

    private static byte[] ToBytes(IReadOnlyList<float>? values)
    {
        if (values == null || values.Count == 0)
        {
            return Array.Empty<byte>();
        }

        float[] floats = values.ToArray();
        byte[] bytes = new byte[checked(floats.Length * sizeof(float))];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static IReadOnlyList<OnnxEngineRuntimeOutputArtifact> CreateLegacyOutputTensors(
        string tensorName,
        IReadOnlyList<int> shape,
        int outputElementCount,
        IReadOnlyList<float> outputPreview,
        byte[] rawOutputBytes)
    {
        if (outputElementCount <= 0 && (rawOutputBytes == null || rawOutputBytes.Length == 0))
        {
            return Array.Empty<OnnxEngineRuntimeOutputArtifact>();
        }

        return new[]
        {
            new OnnxEngineRuntimeOutputArtifact(
                tensorName,
                shape,
                outputElementCount,
                outputPreview,
                rawOutputBytes)
        };
    }

    private static byte[] CombineRawOutputs(IReadOnlyList<OnnxEngineRuntimeOutputArtifact> outputTensors)
    {
        int length = checked(outputTensors.Sum(static output => output.RawBytes.Length));
        byte[] combined = new byte[length];
        int offset = 0;
        foreach (OnnxEngineRuntimeOutputArtifact output in outputTensors)
        {
            Buffer.BlockCopy(output.RawBytes, 0, combined, offset, output.RawBytes.Length);
            offset += output.RawBytes.Length;
        }

        return combined;
    }
}
