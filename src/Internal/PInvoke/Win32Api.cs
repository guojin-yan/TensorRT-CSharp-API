using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Internal.PInvoke
{
    /// <summary>
    /// Win32API Wrapper
    /// </summary>
    public static class Win32Api
    {

#if DOTNET_FRAMEWORK
        private const CharSet DefaultCharSet = CharSet.Auto;
#else
        private const CharSet DefaultCharSet = CharSet.Unicode;
#endif

        [Pure, DllImport("kernel32", CallingConvention = CallingConvention.Winapi,
             SetLastError = true, CharSet = DefaultCharSet, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        public static extern IntPtr LoadLibrary(string dllPath);
    }
}
