using System.Collections.Generic;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Linq;
using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Internal.PInvoke;
using JYPPX.TensorRtSharpv.Internal.PInvoke;



#if DOTNET_FRAMEWORK
using System.Security;
using System.Security.Permissions;
#endif

// TODO
#pragma warning disable CA5393
[assembly: DefaultDllImportSearchPaths(DllImportSearchPath.LegacyBehavior)]

// ReSharper disable InconsistentNaming
#pragma warning disable 1591
#pragma warning disable CA1805 // Do not initialize unnecessarily.


namespace JYPPX.TensorRtSharp.ExternalInterface
{

#if DOTNET_FRAMEWORK
    [SuppressUnmanagedCodeSecurity]
#endif

    /// <summary>
    /// Introducing C API.
    /// </summary>
    public static partial class NativeMethods
    {
        private const string dllExtern = "TensorRT-C-API";



        private const UnmanagedType StringUnmanagedTypeWindows = UnmanagedType.LPStr;

        private const UnmanagedType StringUnmanagedTypeNotWindows =
#if  NET48 || NET481
            UnmanagedType.LPStr;
#else
            UnmanagedType.LPUTF8Str;
#endif

        /// <summary>
        /// Is tried P/Invoke once
        /// </summary>
        private static bool tried;

        /// <summary>
        /// Static constructor
        /// </summary>
#if DOTNET_FRAMEWORK
        [SecurityPermission(SecurityAction.Demand, Flags = SecurityPermissionFlag.UnmanagedCode)]
#endif
        static NativeMethods()
        {
            LoadLibraries(WindowsLibraryLoader.Instance.AdditionalPaths);

            // call cv to enable redirecting 
            TryPInvoke();
        }

#pragma warning disable CA1801
        public static void HandleException(InitExceptionStatus status)
        {
#if DOTNETCORE
        // Check if there has been an exception
        if (status == ExceptionStatus.Occurred /*&& IsUnix()*/) // thrown can be 1 when unix 
        {
            ExceptionHandler.ThrowPossibleException();
        }
#else
#endif
        }
#pragma warning restore CA1801

        /// <summary>
        /// Load DLL files dynamically using Win32 LoadLibrary
        /// </summary>
        /// <param name="additionalPaths"></param>
        public static void LoadLibraries(IEnumerable<string>? additionalPaths = null)
        {
            if (IsWasm())
            {
                return;
            }

            if (IsUnix())
            {
#if DOTNETCORE
            ExceptionHandler.RegisterExceptionCallback();
#endif
                return;
            }

            var ap = (additionalPaths is null) ? Array.Empty<string>() : additionalPaths.ToArray();


            WindowsLibraryLoader.Instance.LoadLibrary(dllExtern, ap);


        }

        /// <summary>
        /// Checks whether PInvoke functions can be called
        /// </summary>
        public static void TryPInvoke()
        {
#pragma warning disable CA1031
            if (tried)
                return;
            tried = true;

            try
            {
                var ret = InitNvinfer();
                GC.KeepAlive(ret);
            }
            catch (DllNotFoundException e)
            {
                var exception = PInvokeHelper.CreateException(e);
                try { Console.WriteLine(exception.Message); }
                // ReSharper disable once EmptyGeneralCatchClause
                catch { }
                try { Debug.WriteLine(exception.Message); }
                // ReSharper disable once EmptyGeneralCatchClause
                catch { }
                throw exception;
            }
            catch (BadImageFormatException e)
            {
                var exception = PInvokeHelper.CreateException(e);
                try { Console.WriteLine(exception.Message); }
                // ReSharper disable once EmptyGeneralCatchClause
                catch { }
                try { Debug.WriteLine(exception.Message); }
                // ReSharper disable once EmptyGeneralCatchClause
                catch { }
                throw exception;
            }
            catch (Exception e)
            {
                var ex = e.InnerException ?? e;
                try { Console.WriteLine(ex.Message); }
                // ReSharper disable once EmptyGeneralCatchClause
                catch { }
                try { Debug.WriteLine(ex.Message); }
                // ReSharper disable once EmptyGeneralCatchClause
                catch { }
                throw;
            }
#pragma warning restore CA1031
        }

        /// <summary>
        /// Returns whether the OS is Windows or not
        /// </summary>
        /// <returns></returns>
        public static bool IsWindows()
        {
#if NET48
            return !IsUnix();
#else
            return RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
#endif
        }

        /// <summary>
        /// Returns whether the OS is *nix or not
        /// </summary>
        /// <returns></returns>
        public static bool IsUnix()
        {
#if NET48
            var p = Environment.OSVersion.Platform;
            return (p == PlatformID.Unix ||
                    p == PlatformID.MacOSX ||
                    (int)p == 128);
#elif NETCOREAPP3_1_OR_GREATER
            return RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
                   RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ||
                   RuntimeInformation.IsOSPlatform(OSPlatform.FreeBSD);
#else
            return RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
                RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
#endif
        }

        /// <summary>
        /// Returns whether the runtime is Mono or not
        /// </summary>
        /// <returns></returns>
        public static bool IsMono()
        {
            return (Type.GetType("Mono.Runtime") is not null);
        }

        /// <summary>
        /// Returns whether the architecture is Wasm or not
        /// </summary>
        /// <returns></returns>
        public static bool IsWasm()
        {
#if NET6_0
        return RuntimeInformation.OSArchitecture == Architecture.Wasm;
#else
            return false;
#endif
        }

    }
}
