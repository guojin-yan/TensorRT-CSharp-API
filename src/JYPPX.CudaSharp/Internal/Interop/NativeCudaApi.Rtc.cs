using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    private const ulong MaximumRtcCopiedOutputSize = 512UL * 1024UL * 1024UL;
    private delegate BridgeStatusCode RtcUtf8BufferGetter(byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);

    public static NativeCudaRtcCapabilityInfo QueryRtcCapability()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_query_capability_safe(out NativeCudaRtcCapabilityInfo info));
        return info;
    }

    public static string GetRtcLoadedLibraryName()
    {
        return ReadRtcUtf8Buffer(NativeMethodsCuda.jyppx_cuda_rtc_get_loaded_library_name_safe);
    }

    public static string GetRtcDependencyDiagnostic()
    {
        return ReadRtcUtf8Buffer(NativeMethodsCuda.jyppx_cuda_rtc_get_dependency_diagnostic_safe);
    }

    public static SafeCudaRtcProgramHandle CreateRtcProgram(CudaRtcProgramSource source)
    {
        byte[] sourceBytes = Encoding.UTF8.GetBytes(source.Source);
        byte[] programNameBytes = Encoding.UTF8.GetBytes(source.ProgramName);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_program_create_safe(
            sourceBytes,
            SizeOf(sourceBytes.Length),
            programNameBytes,
            SizeOf(programNameBytes.Length),
            out SafeCudaRtcProgramHandle program));

        try
        {
            foreach (CudaRtcHeader header in source.Headers)
            {
                byte[] headerBytes = Encoding.UTF8.GetBytes(header.Source);
                byte[] includeNameBytes = Encoding.UTF8.GetBytes(header.IncludeName);
                CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_program_add_header_safe(
                    program,
                    headerBytes,
                    SizeOf(headerBytes.Length),
                    includeNameBytes,
                    SizeOf(includeNameBytes.Length)));
            }

            foreach (string expression in source.NameExpressions)
            {
                byte[] expressionBytes = Encoding.UTF8.GetBytes(expression);
                CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_program_add_name_expression_safe(
                    program,
                    expressionBytes,
                    SizeOf(expressionBytes.Length)));
            }
            return program;
        }
        catch
        {
            program.Dispose();
            throw;
        }
    }

    public static int CompileRtcProgram(SafeCudaRtcProgramHandle program, IReadOnlyList<string> options)
    {
        using Utf8StringArrayScope optionScope = new Utf8StringArrayScope(options);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_program_compile_safe(
            program,
            optionScope.Pointer,
            checked((uint)options.Count),
            out int compilerResult));
        return compilerResult;
    }

    public static string GetRtcProgramLog(SafeCudaRtcProgramHandle program)
    {
        return ReadRtcUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
                NativeMethodsCuda.jyppx_cuda_rtc_program_get_log_safe(program, buffer, size, out required));
    }

    public static byte[]? TryCopyRtcArtifact(SafeCudaRtcProgramHandle program, CudaRtcArtifactKind kind)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_program_try_get_artifact_size_safe(
            program,
            (int)kind,
            out int available,
            out UIntPtr nativeSize));
        if (available == 0)
        {
            return null;
        }

        byte[] bytes = AllocateRtcBuffer(nativeSize, "CUDA RTC artifact");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_rtc_program_copy_artifact_safe(
            program,
            (int)kind,
            bytes,
            SizeOf(bytes.Length),
            out UIntPtr writtenSize));
        ValidateCopiedSize(bytes, writtenSize, "CUDA RTC artifact");
        return bytes;
    }

    public static string GetRtcLoweredName(SafeCudaRtcProgramHandle program, uint expressionIndex)
    {
        return ReadRtcUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
                NativeMethodsCuda.jyppx_cuda_rtc_program_get_lowered_name_safe(program, expressionIndex, buffer, size, out required));
    }

    private static string ReadRtcUtf8Buffer(RtcUtf8BufferGetter getter)
    {
        CudaNativeStatus.ThrowIfFailed(getter(Array.Empty<byte>(), UIntPtr.Zero, out UIntPtr requiredSize));
        byte[] buffer = AllocateRtcBuffer(requiredSize, "CUDA RTC UTF-8 output");
        CudaNativeStatus.ThrowIfFailed(getter(buffer, SizeOf(buffer.Length), out UIntPtr copiedSize));
        ValidateCopiedSize(buffer, copiedSize, "CUDA RTC UTF-8 output");
        int terminator = Array.IndexOf(buffer, (byte)0);
        int length = terminator >= 0 ? terminator : buffer.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(buffer, 0, length);
    }

    private static byte[] AllocateRtcBuffer(UIntPtr nativeSize, string description)
    {
        ulong size = nativeSize.ToUInt64();
        if (size == 0 || size > MaximumRtcCopiedOutputSize || size > int.MaxValue)
        {
            throw new InvalidOperationException($"{description} reported an invalid copied size: {size}.");
        }
        return new byte[checked((int)size)];
    }

    private static void ValidateCopiedSize(byte[] bytes, UIntPtr copiedSize, string description)
    {
        if (copiedSize.ToUInt64() != (ulong)bytes.Length)
        {
            throw new InvalidOperationException(
                $"{description} size changed between query and copy: expected {bytes.Length}, copied {copiedSize.ToUInt64()}.");
        }
    }

    private static UIntPtr SizeOf(int length)
    {
        return new UIntPtr(checked((uint)length));
    }

    private sealed class Utf8StringArrayScope : IDisposable
    {
        private readonly List<IntPtr> _strings = new List<IntPtr>();

        public Utf8StringArrayScope(IReadOnlyList<string> values)
        {
            if (values.Count == 0)
            {
                Pointer = IntPtr.Zero;
                return;
            }

            Pointer = Marshal.AllocHGlobal(checked(values.Count * IntPtr.Size));
            try
            {
                for (int index = 0; index < values.Count; index++)
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(values[index]);
                    IntPtr value = Marshal.AllocHGlobal(bytes.Length + 1);
                    Marshal.Copy(bytes, 0, value, bytes.Length);
                    Marshal.WriteByte(value, bytes.Length, 0);
                    _strings.Add(value);
                    Marshal.WriteIntPtr(Pointer, index * IntPtr.Size, value);
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public IntPtr Pointer { get; private set; }

        public void Dispose()
        {
            foreach (IntPtr value in _strings)
            {
                Marshal.FreeHGlobal(value);
            }
            _strings.Clear();
            if (Pointer != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(Pointer);
                Pointer = IntPtr.Zero;
            }
        }
    }
}
