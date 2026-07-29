using System;
using System.IO;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class MnistOnnxRuntimeService
{
    private static void ValidateOptions(MnistOnnxRuntimeOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.OnnxPath) || !File.Exists(options.OnnxPath))
        {
            throw new FileNotFoundException("MNIST ONNX model was not found.", options.OnnxPath);
        }

        if (string.IsNullOrWhiteSpace(options.InputPgmPath) || !File.Exists(options.InputPgmPath))
        {
            throw new FileNotFoundException("MNIST PGM input was not found.", options.InputPgmPath);
        }

        if (options.ExpectedDigit < 0 || options.ExpectedDigit > 9)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ExpectedDigit), "Expected digit must be in [0, 9].");
        }

        if (options.WorkspaceBytes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.WorkspaceBytes), "Workspace size must be greater than zero.");
        }

        if (options.MinimumConfidence <= 0.0f || options.MinimumConfidence > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MinimumConfidence), "Minimum confidence must be in (0, 1].");
        }
    }
}
