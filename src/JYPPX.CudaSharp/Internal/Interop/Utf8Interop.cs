using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static class Utf8Interop
{
    public static Utf8StringScope ToNativeString(string? value)
    {
        return new Utf8StringScope(value);
    }

    public static string ReadString(IntPtr value)
    {
        if (value == IntPtr.Zero)
        {
            return string.Empty;
        }

#if NET5_0_OR_GREATER
        return Marshal.PtrToStringUTF8(value) ?? string.Empty;
#else
        List<byte> bytes = new List<byte>();
        int offset = 0;
        while (true)
        {
            byte current = Marshal.ReadByte(value, offset);
            if (current == 0)
            {
                break;
            }

            bytes.Add(current);
            offset++;
        }

        return bytes.Count == 0 ? string.Empty : Encoding.UTF8.GetString(bytes.ToArray(), 0, bytes.Count);
#endif
    }

    internal sealed class Utf8StringScope : IDisposable
    {
        public Utf8StringScope(string? value)
        {
            if (value == null)
            {
                Pointer = IntPtr.Zero;
                return;
            }

            byte[] bytes = Encoding.UTF8.GetBytes(value);
            Pointer = Marshal.AllocHGlobal(bytes.Length + 1);
            Marshal.Copy(bytes, 0, Pointer, bytes.Length);
            Marshal.WriteByte(Pointer, bytes.Length, 0);
        }

        public IntPtr Pointer { get; }

        public void Dispose()
        {
            if (Pointer != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(Pointer);
            }
        }
    }
}
