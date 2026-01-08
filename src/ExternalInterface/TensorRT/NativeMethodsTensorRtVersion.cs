using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtVersion_GetVersionMajor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtVersion_GetVersionMajor();

        [Pure, DllImport(dllExtern, EntryPoint = "trtVersion_GetVersionMinor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtVersion_GetVersionMinor();


        [Pure, DllImport(dllExtern, EntryPoint = "trtVersion_GetVersionPatch",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtVersion_GetVersionPatch();
    }
 }
