using System;
using System.Collections.Generic;
using System.Linq;

namespace YoloVisionSample;

public sealed class YoloRuntimeOutputSet
{
    public YoloRuntimeOutputSet(IEnumerable<YoloRuntimeOutputTensor> outputs)
    {
        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs));
        }

        Outputs = outputs.ToArray();
        if (Outputs.Count == 0)
        {
            throw new ArgumentException("At least one YOLO output tensor is required.", nameof(outputs));
        }
    }

    public IReadOnlyList<YoloRuntimeOutputTensor> Outputs { get; }

    public YoloRuntimeOutputTensor GetRequired(YoloOutputTensorRole role)
    {
        YoloRuntimeOutputTensor? output = Outputs.FirstOrDefault(item => item.Role == role);
        if (output == null)
        {
            throw new ArgumentException($"Required YOLO output role '{role}' was not provided.");
        }

        return output;
    }

    public YoloRuntimeOutputTensor? TryGet(YoloOutputTensorRole role)
    {
        return Outputs.FirstOrDefault(item => item.Role == role);
    }

    public override string ToString()
    {
        return string.Join("; ", Outputs.Select(static output => output.ToString()));
    }
}
