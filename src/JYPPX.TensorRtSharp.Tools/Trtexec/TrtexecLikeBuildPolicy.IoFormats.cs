using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
{
    private static void ApplyIoFormats(
        TensorRtNetworkDefinition network,
        string specification,
        bool isInput,
        List<string> log)
    {
        if (string.IsNullOrWhiteSpace(specification))
        {
            return;
        }

        string policyName = isInput ? "InputIOFormats" : "OutputIOFormats";
        string optionName = isInput ? "--inputIOFormats" : "--outputIOFormats";
        IReadOnlyList<TrtexecLikeIoFormatSpec> specs = ParseIoFormats(specification, optionName);
        int tensorCount = isInput ? network.InputCount : network.OutputCount;
        if (tensorCount == 0)
        {
            throw new InvalidOperationException(optionName + " cannot be applied because the parsed network has no matching I/O tensors.");
        }

        if (specs.Count != 1 && specs.Count != tensorCount)
        {
            throw new InvalidOperationException($"{optionName} requires one broadcast specification or exactly {tensorCount} specifications.");
        }

        List<TensorRtTensor> tensors = new List<TensorRtTensor>(tensorCount);
        try
        {
            for (int index = 0; index < tensorCount; index++)
            {
                tensors.Add(isInput ? network.GetInput(index) : network.GetOutput(index));
            }

            for (int index = 0; index < tensors.Count; index++)
            {
                TensorRtDataType requestedType = specs.Count == 1 ? specs[0].DataType : specs[index].DataType;
                if (!IsDataTypeSupported(network.Line, requestedType))
                {
                    log.Add($"TrtexecBuildPolicy Name={policyName} Applied=False Requested={specification} TensorIndex={index} VersionGuard={network.Line} Reason=data-type-not-supported-on-api-line ReadbackMatch=False");
                    return;
                }

                if (network.Line == TensorRtApiLine.TensorRt11 && tensors[index].DataType != requestedType)
                {
                    log.Add($"TrtexecBuildPolicy Name={policyName} Applied=False Requested={specification} TensorIndex={index} RequestedType={requestedType} ReadbackType={tensors[index].DataType} VersionGuard=TRT11 Reason=tensor-set-type-removed ReadbackMatch=False");
                    return;
                }
            }

            bool readbackMatch = true;
            for (int index = 0; index < tensors.Count; index++)
            {
                TrtexecLikeIoFormatSpec spec = specs.Count == 1 ? specs[0] : specs[index];
                TensorRtTensor tensor = tensors[index];
                if (network.Line != TensorRtApiLine.TensorRt11)
                {
                    tensor.DataType = spec.DataType;
                }

                tensor.AllowedFormats = spec.Formats;
                readbackMatch &= tensor.DataType == spec.DataType && tensor.AllowedFormats == spec.Formats;
            }

            log.Add($"TrtexecBuildPolicy Name={policyName} Applied={readbackMatch} Requested={specification} TensorCount={tensorCount} Broadcast={specs.Count == 1} TypeMode={(network.Line == TensorRtApiLine.TensorRt11 ? "validated-existing" : "set-and-readback")} ReadbackMatch={readbackMatch}");
            if (!readbackMatch)
            {
                throw new InvalidOperationException(policyName + " did not match TensorRT tensor readback.");
            }
        }
        finally
        {
            foreach (TensorRtTensor tensor in tensors)
            {
                tensor.Dispose();
            }
        }
    }
}
