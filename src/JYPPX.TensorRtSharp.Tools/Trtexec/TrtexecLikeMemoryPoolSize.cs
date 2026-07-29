using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class TrtexecLikeMemoryPoolSize
{
    public TrtexecLikeMemoryPoolSize(string name, ulong sizeMiB)
    {
        Name = string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("Memory pool name is required.", nameof(name)) : name.Trim();
        SizeMiB = sizeMiB;
        SizeBytes = checked(sizeMiB * 1024UL * 1024UL);
    }

    public string Name { get; }

    public ulong SizeMiB { get; }

    public ulong SizeBytes { get; }

    /// <summary>
    /// Resolves the trtexec pool token to the typed TensorRT memory-pool enum.
    /// 将 trtexec memory-pool token 解析为强类型 TensorRT memory-pool 枚举。
    /// </summary>
    public TensorRtMemoryPoolType ToTensorRtMemoryPoolType()
    {
        string normalized = Name
            .Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace(" ", string.Empty, StringComparison.Ordinal)
            .ToLowerInvariant();

        return normalized switch
        {
            "workspace" => TensorRtMemoryPoolType.Workspace,
            "dlasram" or "dlamanagedsram" => TensorRtMemoryPoolType.DlaManagedSram,
            "dlalocaldram" => TensorRtMemoryPoolType.DlaLocalDram,
            "dlaglobaldram" => TensorRtMemoryPoolType.DlaGlobalDram,
            "tacticdram" => TensorRtMemoryPoolType.TacticDram,
            "tacticsharedmem" or "tacticsharedmemory" => TensorRtMemoryPoolType.TacticSharedMemory,
            _ => throw new ArgumentException(
                $"Unsupported TensorRT memory pool '{Name}'. Supported pools: workspace, dlaSRAM, dlaLocalDRAM, dlaGlobalDRAM, tacticDRAM, tacticSharedMem.",
                nameof(Name))
        };
    }

    public string ToArgumentSegment()
    {
        return Name + ":" + SizeMiB.ToString(CultureInfo.InvariantCulture);
    }
}
