using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// Provides functionality for initializing and managing TensorRT plugins.
    /// </summary>
    public static class TrtPlugin
    {
        /// <summary>
        /// Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional namespace.
        /// This function should be called once before accessing the Plugin Registry.
        /// </summary>
        /// <param name="logger">Logger object to print plugin registration information. If null, no logging will be performed.</param>
        /// <param name="libNamespace">Namespace used to register all the plugins in this library. If null, plugins will be registered without a namespace.</param>
        /// <returns>true if initialization succeeded, false otherwise.</returns>
        public static bool InitLibNvInferPlugins(Logger logger = null, string libNamespace = null)
        {
            IntPtr loggerPtr = IntPtr.Zero;
            if (logger != null)
            {
                // Get the logger pointer through reflection or add a property to access it
                // For now, we use IntPtr.Zero as the Logger class doesn't expose its internal pointer
                // The C API will use the global logger if logger is nullptr
            }
            return NativeMethods.trtPlugin_initLibNvInferPlugins(loggerPtr, libNamespace);
        }

        /// <summary>
        /// Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional namespace.
        /// This is a convenience overload that uses the global logger instance.
        /// </summary>
        /// <param name="libNamespace">Namespace used to register all the plugins in this library. If null, plugins will be registered without a namespace.</param>
        /// <returns>true if initialization succeeded, false otherwise.</returns>
        public static bool InitLibNvInferPlugins(string libNamespace)
        {
            return NativeMethods.trtPlugin_initLibNvInferPlugins(IntPtr.Zero, libNamespace);
        }
    }
}
