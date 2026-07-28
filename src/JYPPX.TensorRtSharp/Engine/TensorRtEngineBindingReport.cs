using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Summarizes TensorRT engine I/O metadata and optional execution-context readiness for deployment.
/// 汇总 TensorRT engine I/O 元数据以及可选 execution-context 就绪状态，用于模型部署诊断。
/// </summary>
public sealed class TensorRtEngineBindingReport
{
    /// <summary>
    /// Creates an engine binding report.
    /// 创建 engine 绑定报告。
    /// </summary>
    public TensorRtEngineBindingReport(
        string engineName,
        int profileIndex,
        IReadOnlyList<TensorRtEngineTensorBinding> tensors,
        TensorRtExecutionContextReadiness? readiness)
    {
        EngineName = engineName;
        ProfileIndex = profileIndex;
        Tensors = tensors;
        Readiness = readiness;
    }

    /// <summary>
    /// Gets the TensorRT engine name.
    /// 获取 TensorRT engine 名称。
    /// </summary>
    public string EngineName { get; }

    /// <summary>
    /// Gets the optimization profile index used to query profile-specific tensor metadata.
    /// 获取用于查询 profile 相关 tensor 元数据的 optimization profile 索引。
    /// </summary>
    public int ProfileIndex { get; }

    /// <summary>
    /// Gets the engine I/O tensor binding snapshots.
    /// 获取 engine I/O tensor 绑定快照集合。
    /// </summary>
    public IReadOnlyList<TensorRtEngineTensorBinding> Tensors { get; }

    /// <summary>
    /// Gets the optional execution-context readiness snapshot.
    /// 获取可选的 execution context 就绪状态快照。
    /// </summary>
    public TensorRtExecutionContextReadiness? Readiness { get; }

    /// <summary>
    /// Gets whether an execution-context snapshot is attached and ready for enqueue.
    /// 获取是否附加了 execution context 快照且该快照已可执行 enqueue。
    /// </summary>
    public bool IsReadyForEnqueue => Readiness != null && Readiness.IsReadyForEnqueue;
    /// <summary>
    /// Finds a tensor binding by name.
    /// 按名称查找 tensor 绑定信息。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <returns>The matching tensor binding. 匹配的 tensor 绑定信息。</returns>
    public TensorRtEngineTensorBinding GetTensor(string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        foreach (TensorRtEngineTensorBinding tensor in Tensors)
        {
            if (StringComparer.Ordinal.Equals(tensor.Name, tensorName))
            {
                return tensor;
            }
        }

        throw new ArgumentException($"Tensor '{tensorName}' was not found in this binding report.", nameof(tensorName));
    }

    /// <summary>
    /// Gets all input tensor bindings.
    /// 获取所有输入 tensor 绑定信息。
    /// </summary>
    /// <returns>The input tensor bindings. 输入 tensor 绑定信息。</returns>
    public IReadOnlyList<TensorRtEngineTensorBinding> GetInputs()
    {
        return GetByMode(TensorRtIOMode.Input);
    }

    /// <summary>
    /// Gets all output tensor bindings.
    /// 获取所有输出 tensor 绑定信息。
    /// </summary>
    /// <returns>The output tensor bindings. 输出 tensor 绑定信息。</returns>
    public IReadOnlyList<TensorRtEngineTensorBinding> GetOutputs()
    {
        return GetByMode(TensorRtIOMode.Output);
    }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    /// <returns>A summary line. 摘要文本。</returns>
    public override string ToString()
    {
        return $"{EngineName} profile={ProfileIndex} tensors={Tensors.Count} ready={IsReadyForEnqueue}";
    }

    private IReadOnlyList<TensorRtEngineTensorBinding> GetByMode(TensorRtIOMode mode)
    {
        List<TensorRtEngineTensorBinding> tensors = new List<TensorRtEngineTensorBinding>();
        foreach (TensorRtEngineTensorBinding tensor in Tensors)
        {
            if (tensor.IOMode == mode)
            {
                tensors.Add(tensor);
            }
        }

        return tensors;
    }
}
