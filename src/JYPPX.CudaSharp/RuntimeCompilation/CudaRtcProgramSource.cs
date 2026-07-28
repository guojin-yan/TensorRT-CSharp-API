using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;

namespace JYPPX.CudaSharp;

/// <summary>Immutable virtual header supplied to NVRTC. 提供给 NVRTC 的不可变虚拟 header。</summary>
public sealed class CudaRtcHeader
{
    /// <summary>Creates a copied virtual header. 创建复制型虚拟 header。</summary>
    public CudaRtcHeader(string includeName, string source)
    {
        IncludeName = CudaRtcValidation.ValidateText(includeName, nameof(includeName), 4096, false);
        Source = CudaRtcValidation.ValidateText(source, nameof(source), 4 * 1024 * 1024, true);
    }

    /// <summary>Gets the virtual include name. 获取虚拟 include 名称。</summary>
    public string IncludeName { get; }

    /// <summary>Gets the copied header source. 获取复制的 header 源码。</summary>
    public string Source { get; }
}

/// <summary>Immutable NVRTC source, virtual headers, and name expressions. 不可变的 NVRTC 源码、虚拟 header 与 name expression。</summary>
public sealed class CudaRtcProgramSource
{
    private readonly ReadOnlyCollection<CudaRtcHeader> _headers;
    private readonly ReadOnlyCollection<string> _nameExpressions;

    /// <summary>Creates an immutable program-source snapshot. 创建不可变 program source 快照。</summary>
    public CudaRtcProgramSource(
        string source,
        string programName = "program.cu",
        IEnumerable<CudaRtcHeader>? headers = null,
        IEnumerable<string>? nameExpressions = null)
    {
        Source = CudaRtcValidation.ValidateText(source, nameof(source), 16 * 1024 * 1024, false);
        ProgramName = CudaRtcValidation.ValidateText(programName, nameof(programName), 4096, false);

        var headerCopy = new List<CudaRtcHeader>();
        var includeNames = new HashSet<string>(StringComparer.Ordinal);
        int combinedHeaderBytes = 0;
        if (headers != null)
        {
            foreach (CudaRtcHeader? header in headers)
            {
                if (header == null)
                {
                    throw new ArgumentException("CUDA RTC headers must not contain null entries.", nameof(headers));
                }
                if (!includeNames.Add(header.IncludeName))
                {
                    throw new ArgumentException($"Duplicate CUDA RTC include name: {header.IncludeName}.", nameof(headers));
                }
                combinedHeaderBytes = checked(combinedHeaderBytes + Encoding.UTF8.GetByteCount(header.Source));
                if (combinedHeaderBytes > 16 * 1024 * 1024 || headerCopy.Count >= 256)
                {
                    throw new ArgumentException("CUDA RTC headers exceed the bounded count or combined byte limit.", nameof(headers));
                }
                headerCopy.Add(header);
            }
        }

        var expressionCopy = new List<string>();
        var uniqueExpressions = new HashSet<string>(StringComparer.Ordinal);
        if (nameExpressions != null)
        {
            foreach (string? expression in nameExpressions)
            {
                string validated = CudaRtcValidation.ValidateText(expression, nameof(nameExpressions), 4096, false);
                if (!uniqueExpressions.Add(validated))
                {
                    throw new ArgumentException($"Duplicate CUDA RTC name expression: {validated}.", nameof(nameExpressions));
                }
                if (expressionCopy.Count >= 256)
                {
                    throw new ArgumentException("CUDA RTC name-expression count exceeds the bounded limit.", nameof(nameExpressions));
                }
                expressionCopy.Add(validated);
            }
        }

        _headers = new ReadOnlyCollection<CudaRtcHeader>(headerCopy);
        _nameExpressions = new ReadOnlyCollection<string>(expressionCopy);
    }

    /// <summary>Gets the copied CUDA C++ source. 获取复制的 CUDA C++ 源码。</summary>
    public string Source { get; }

    /// <summary>Gets the virtual program name used in diagnostics. 获取诊断中使用的虚拟 program 名称。</summary>
    public string ProgramName { get; }

    /// <summary>Gets the copied virtual headers in stable order. 获取稳定顺序的虚拟 header 副本。</summary>
    public IReadOnlyList<CudaRtcHeader> Headers => _headers;

    /// <summary>Gets name expressions in stable order. 获取稳定顺序的 name expression。</summary>
    public IReadOnlyList<string> NameExpressions => _nameExpressions;
}

internal static class CudaRtcValidation
{
    private static readonly Encoding StrictUtf8 = new UTF8Encoding(false, true);

    public static string ValidateText(string? value, string parameterName, int maximumUtf8Bytes, bool allowEmpty)
    {
        if (value == null)
        {
            throw new ArgumentNullException(parameterName);
        }
        if (!allowEmpty && value.Length == 0)
        {
            throw new ArgumentException("CUDA RTC text must not be empty.", parameterName);
        }
        if (value.IndexOf('\0') >= 0)
        {
            throw new ArgumentException("CUDA RTC text must not contain an embedded NUL.", parameterName);
        }
        int utf8ByteCount;
        try
        {
            utf8ByteCount = StrictUtf8.GetByteCount(value);
        }
        catch (EncoderFallbackException exception)
        {
            throw new ArgumentException("CUDA RTC text must be valid UTF-8 text without unpaired surrogate code units.", parameterName, exception);
        }
        if (utf8ByteCount > maximumUtf8Bytes)
        {
            throw new ArgumentException("CUDA RTC text exceeds the bounded UTF-8 byte limit.", parameterName);
        }
        return value;
    }
}
