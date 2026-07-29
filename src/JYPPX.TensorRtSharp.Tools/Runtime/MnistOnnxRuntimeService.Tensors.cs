using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class MnistOnnxRuntimeService
{
    private static TensorRtEngineTensorBinding SingleTensor(
        IReadOnlyList<TensorRtEngineTensorBinding> tensors,
        string role)
    {
        if (tensors.Count != 1)
        {
            throw new InvalidOperationException($"MNIST runtime expects exactly one {role} tensor. Actual={tensors.Count}.");
        }

        return tensors[0];
    }

    private static void EnsureFloatDeviceTensor(TensorRtEngineTensorBinding tensor, string role)
    {
        if (tensor.DataType != TensorRtDataType.Float)
        {
            throw new NotSupportedException($"MNIST {role} tensor must use Float. Actual={tensor.DataType}.");
        }

        if (tensor.Location != TensorRtTensorLocation.Device)
        {
            throw new NotSupportedException($"MNIST {role} tensor must use device memory. Actual={tensor.Location}.");
        }
    }

    private static TensorRtDims ResolveShape(
        TensorRtExecutionContext context,
        TensorRtEngineTensorBinding tensor)
    {
        TensorRtDims contextShape = context.GetTensorShape(tensor.Name);
        if (IsConcrete(contextShape))
        {
            return contextShape;
        }

        if (IsConcrete(tensor.EngineShape))
        {
            return tensor.EngineShape;
        }

        throw new InvalidOperationException($"MNIST tensor '{tensor.Name}' does not have a concrete shape.");
    }

    private static bool IsConcrete(TensorRtDims shape)
    {
        return shape.Values.Length > 0 && shape.Values.All(static value => value > 0);
    }

    private static int ElementCount(TensorRtDims shape)
    {
        int count = 1;
        foreach (int value in shape.Values)
        {
            count = checked(count * value);
        }

        return count;
    }
}
