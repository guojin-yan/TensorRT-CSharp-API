using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct InterfaceInfo
    {
        [MarshalAs(UnmanagedType.LPStr)]
        public string kind;
        public int major; 
        public int minor;
    }
}
