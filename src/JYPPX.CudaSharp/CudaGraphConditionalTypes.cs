using System;

namespace JYPPX.CudaSharp;

/// <summary>Identifies the CUDA conditional graph node body policy. 标识 CUDA conditional graph 节点的 body 策略。</summary>
public enum CudaGraphConditionalNodeType
{
    /// <summary>Executes the first or optional second body as an if/else branch. 以 if/else 分支执行第一个或可选的第二个 body。</summary>
    If = 0,
    /// <summary>Repeats the body while the condition remains nonzero. 条件保持非零时重复执行 body。</summary>
    While = 1,
    /// <summary>Selects one body using the condition value. 使用条件值选择一个 body。</summary>
    Switch = 2
}

/// <summary>Controls initialization of a CUDA conditional value at graph launch. 控制 graph 启动时 CUDA 条件值的初始化。</summary>
[Flags]
public enum CudaGraphConditionalHandleFlags : uint
{
    /// <summary>Does not request default-value assignment. 不请求分配默认值。</summary>
    None = 0,
    /// <summary>Assigns the configured default value at graph launch. 在 graph 启动时分配已配置的默认值。</summary>
    AssignDefault = 1
}
