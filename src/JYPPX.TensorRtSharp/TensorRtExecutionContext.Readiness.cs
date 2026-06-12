using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Creates a deployment readiness snapshot for this execution context and engine.
    /// 为当前 execution context 与 engine 创建部署就绪状态快照。
    /// </summary>
    /// <param name="engine">The engine that created this context. 创建该 context 的 engine。</param>
    /// <param name="runShapeInference">True to call TensorRT inferShapes before reporting missing-shape count. 为 true 时先调用 TensorRT inferShapes 并报告缺失 shape 数量。</param>
    /// <returns>A readiness snapshot suitable for diagnostics before enqueue. 可用于 enqueue 前诊断的就绪状态快照。</returns>
    public TensorRtExecutionContextReadiness GetReadiness(TensorRtEngine engine, bool runShapeInference = false)
    {
        if (engine == null)
        {
            throw new ArgumentNullException(nameof(engine));
        }

        if (engine.Line != Line)
        {
            throw new ArgumentException("Engine and execution context must belong to the same TensorRT API line.", nameof(engine));
        }

        int? missingShapeCount = runShapeInference ? InferShapes() : (int?)null;
        List<TensorRtTensorBindingState> tensors = new List<TensorRtTensorBindingState>();
        foreach (TensorRtTensorInfo tensor in engine.GetIOTensors())
        {
            TensorRtDims? contextShape = null;
            TensorRtDims? contextStrides = null;
            bool isAddressBound = false;
            long? maxOutputSize = null;
            string? diagnostic = null;

            try
            {
                contextShape = GetTensorShape(tensor.Name);
                contextStrides = GetTensorStrides(tensor.Name);
                isAddressBound = IsTensorAddressBound(tensor.Name);
                if (tensor.IOMode == TensorRtIOMode.Output)
                {
                    maxOutputSize = GetMaxOutputSize(tensor.Name);
                }
            }
            catch (Exception ex)
            {
                diagnostic = ex.Message;
            }

            tensors.Add(new TensorRtTensorBindingState(
                tensor.Index,
                tensor.Name,
                tensor.DataType,
                tensor.IOMode,
                tensor.Shape,
                contextShape,
                contextStrides,
                isAddressBound,
                maxOutputSize,
                diagnostic));
        }

        bool allInputDimensionsSpecified = AllInputDimensionsSpecified;
        bool allInputShapesSpecified = GetAllInputShapesSpecifiedForReadiness(allInputDimensionsSpecified, missingShapeCount);

        return new TensorRtExecutionContextReadiness(
            allInputDimensionsSpecified,
            allInputShapesSpecified,
            OptimizationProfileIndex,
            missingShapeCount,
            tensors);
    }

    private bool GetAllInputShapesSpecifiedForReadiness(bool allInputDimensionsSpecified, int? missingShapeCount)
    {
        try
        {
            return AllInputShapesSpecified;
        }
        catch (BridgeProbeException exception) when (Line == TensorRtApiLine.TensorRt11 && exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            if (missingShapeCount.HasValue)
            {
                return missingShapeCount.Value == 0;
            }

            return allInputDimensionsSpecified;
        }
    }
}
