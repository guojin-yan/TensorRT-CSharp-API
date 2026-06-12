using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static class Utf8Interop
{
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
}

