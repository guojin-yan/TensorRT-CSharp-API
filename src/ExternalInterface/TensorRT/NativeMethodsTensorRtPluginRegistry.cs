using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        #region Deprecated IPluginCreator API (for backward compatibility)

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_registerCreator",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_registerCreator(
            IntPtr registry,
            IntPtr creator,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginNamespace);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_deregisterCreator",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_deregisterCreator(
            IntPtr registry,
            IntPtr creator);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_getPluginCreator",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_getPluginCreator(
            IntPtr registry,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginName,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginVersion,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginNamespace,
            out IntPtr creator);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_getPluginCreatorList",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_getPluginCreatorList(
            IntPtr registry,
            out int numCreators,
            out IntPtr creators);

        #endregion

        #region Error Recorder

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_setErrorRecorder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_setErrorRecorder(
            IntPtr registry,
            IntPtr errorRecorder);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_getErrorRecorder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_getErrorRecorder(
            IntPtr registry,
            out IntPtr errorRecorder);

        #endregion

        #region Parent Search Control

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_isParentSearchEnabled",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_isParentSearchEnabled(
            IntPtr registry,
            out int enabled);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_setParentSearchEnabled",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_setParentSearchEnabled(
            IntPtr registry,
            int enabled);

        #endregion

        #region Library Management

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_loadLibrary",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_loadLibrary(
            IntPtr registry,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginPath,
            out IntPtr handle);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_deregisterLibrary",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_deregisterLibrary(
            IntPtr registry,
            IntPtr handle);

        #endregion

        #region IPluginCreatorInterface API (TensorRT 10.0+)

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_registerCreatorInterface",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_registerCreatorInterface(
            IntPtr registry,
            IntPtr creator,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginNamespace);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_getAllCreatorsInterface",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_getAllCreatorsInterface(
            IntPtr registry,
            out int numCreators,
            out IntPtr creators);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_getCreatorInterface",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_getCreatorInterface(
            IntPtr registry,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginName,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginVersion,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pluginNamespace,
            out IntPtr creator);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_deregisterCreatorInterface",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_deregisterCreatorInterface(
            IntPtr registry,
            IntPtr creator);

        #endregion

        #region Plugin Resource Management

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_acquirePluginResource",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_acquirePluginResource(
            IntPtr registry,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string key,
            IntPtr resource,
            out IntPtr acquiredResource);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_releasePluginResource",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_releasePluginResource(
            IntPtr registry,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string key,
            out int result);

        #endregion

        #region Recursive Creator Query

        [Pure, DllImport(dllExtern, EntryPoint = "trtPluginRegistry_getAllCreatorsRecursive",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPluginRegistry_getAllCreatorsRecursive(
            IntPtr registry,
            out int numCreators,
            out IntPtr creators);

        #endregion
    }
}
