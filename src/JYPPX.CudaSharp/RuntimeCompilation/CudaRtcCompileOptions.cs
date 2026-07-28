using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace JYPPX.CudaSharp;

/// <summary>Immutable validated NVRTC option snapshot. 不可变且已验证的 NVRTC option 快照。</summary>
public sealed class CudaRtcCompileOptions
{
    private readonly ReadOnlyCollection<string> _options;

    /// <summary>Creates a bounded compile-option snapshot. 创建有界 compile option 快照。</summary>
    public CudaRtcCompileOptions(
        string? targetArchitecture = null,
        bool generateLineInfo = false,
        bool deviceDebug = false,
        bool useFastMath = false,
        bool relocatableDeviceCode = false,
        bool emitLtoIr = false,
        IEnumerable<string>? additionalOptions = null)
    {
        TargetArchitecture = ValidateArchitecture(targetArchitecture);
        GenerateLineInfo = generateLineInfo;
        DeviceDebug = deviceDebug;
        UseFastMath = useFastMath;
        RelocatableDeviceCode = relocatableDeviceCode;
        EmitLtoIr = emitLtoIr;

        var options = new List<string>();
        var unique = new HashSet<string>(StringComparer.Ordinal);
        if (TargetArchitecture.Length != 0)
        {
            AddOption(options, unique, "--gpu-architecture=" + TargetArchitecture, nameof(targetArchitecture));
        }
        if (generateLineInfo)
        {
            AddOption(options, unique, "--generate-line-info", nameof(generateLineInfo));
        }
        if (deviceDebug)
        {
            AddOption(options, unique, "--device-debug", nameof(deviceDebug));
        }
        if (useFastMath)
        {
            AddOption(options, unique, "--use_fast_math", nameof(useFastMath));
        }
        if (relocatableDeviceCode)
        {
            AddOption(options, unique, "--relocatable-device-code=true", nameof(relocatableDeviceCode));
        }
        if (emitLtoIr)
        {
            AddOption(options, unique, "--dlink-time-opt", nameof(emitLtoIr));
        }
        if (additionalOptions != null)
        {
            foreach (string? option in additionalOptions)
            {
                AddOption(options, unique, option, nameof(additionalOptions));
            }
        }
        if (options.Count > 256)
        {
            throw new ArgumentException("CUDA RTC option count exceeds the bounded limit.", nameof(additionalOptions));
        }
        _options = new ReadOnlyCollection<string>(options);
    }

    /// <summary>Gets an empty default option snapshot. 获取空的默认 option 快照。</summary>
    public static CudaRtcCompileOptions Default { get; } = new CudaRtcCompileOptions();

    /// <summary>Gets the requested compute or SM architecture. 获取请求的 compute 或 SM architecture。</summary>
    public string TargetArchitecture { get; }

    /// <summary>Gets whether line information is requested. 获取是否请求行号信息。</summary>
    public bool GenerateLineInfo { get; }

    /// <summary>Gets whether device debug output is requested. 获取是否请求 device debug 输出。</summary>
    public bool DeviceDebug { get; }

    /// <summary>Gets whether fast math is enabled. 获取是否启用 fast math。</summary>
    public bool UseFastMath { get; }

    /// <summary>Gets whether relocatable device code is enabled. 获取是否启用 relocatable device code。</summary>
    public bool RelocatableDeviceCode { get; }

    /// <summary>Gets whether LTO IR generation is requested. 获取是否请求 LTO IR。</summary>
    public bool EmitLtoIr { get; }

    /// <summary>Gets the complete retained option list in compiler order. 获取按编译器顺序保留的完整 option 列表。</summary>
    public IReadOnlyList<string> Options => _options;

    private static void AddOption(List<string> options, HashSet<string> unique, string? option, string parameterName)
    {
        string validated = CudaRtcValidation.ValidateText(option, parameterName, 16384, false);
        if (!unique.Add(validated))
        {
            throw new ArgumentException($"Duplicate CUDA RTC option: {validated}.", parameterName);
        }
        options.Add(validated);
    }

    private static string ValidateArchitecture(string? architecture)
    {
        if (string.IsNullOrEmpty(architecture))
        {
            return string.Empty;
        }
        string value = CudaRtcValidation.ValidateText(architecture, nameof(architecture), 64, false);
        bool prefix = value.StartsWith("compute_", StringComparison.Ordinal) || value.StartsWith("sm_", StringComparison.Ordinal);
        int underscore = value.IndexOf('_');
        if (!prefix || underscore < 0 || underscore == value.Length - 1)
        {
            throw new ArgumentException("CUDA RTC target architecture must use compute_XX or sm_XX syntax.", nameof(architecture));
        }
        bool hasDigit = false;
        for (int index = underscore + 1; index < value.Length; index++)
        {
            char character = value[index];
            if (char.IsDigit(character))
            {
                hasDigit = true;
                continue;
            }
            if (!(hasDigit && index == value.Length - 1 && character == 'a'))
            {
                throw new ArgumentException("CUDA RTC target architecture must use compute_XX or sm_XX syntax.", nameof(architecture));
            }
        }
        if (!hasDigit)
        {
            throw new ArgumentException("CUDA RTC target architecture must use compute_XX or sm_XX syntax.", nameof(architecture));
        }
        return value;
    }
}
