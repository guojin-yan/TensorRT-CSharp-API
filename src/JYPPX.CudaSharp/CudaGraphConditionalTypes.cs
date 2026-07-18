using System;

namespace JYPPX.CudaSharp;

/// <summary>Identifies the CUDA conditional graph node body policy.</summary>
public enum CudaGraphConditionalNodeType
{
    If = 0,
    While = 1,
    Switch = 2
}

/// <summary>Controls initialization of a CUDA conditional value at graph launch.</summary>
[Flags]
public enum CudaGraphConditionalHandleFlags : uint
{
    None = 0,
    AssignDefault = 1
}
