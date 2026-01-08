using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.TensorRtSharp.ExternalInterface;
namespace JYPPX.TensorRtSharp.Nvinfer
{
    public class TrtVersion
    {

        #region Public Properties (公共属性)
        /// <summary>
        /// 获取主版本号 (Major Version)
        /// </summary>
        public static int Major
        {
            get { return NativeMethods.trtVersion_GetVersionMajor(); }
        }
        /// <summary>
        /// 获取次版本号 (Minor Version)
        /// </summary>
        public static int Minor
        {
            get { return NativeMethods.trtVersion_GetVersionMinor(); }
        }
        /// <summary>
        /// 获取补丁号 (Patch Version)
        /// </summary>
        public static int Patch
        {
            get { return NativeMethods.trtVersion_GetVersionPatch(); }
        }
        /// <summary>
        /// 获取完整的版本字符串 (例如 "8.5.1")
        /// </summary>
        public static string Version
        {
            get { return string.Format("{0}.{1}.{2}", Major, Minor, Patch); }
        }
        #endregion
        /// <summary>
        /// 返回表示当前对象的字符串。
        /// </summary>
        /// <returns>版本字符串</returns>
        public override string ToString()
        {
            return Version;
        }
    }
}
