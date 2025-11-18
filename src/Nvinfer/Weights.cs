using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    [StructLayout(LayoutKind.Sequential)]
    public struct TrtWeights
    {
        public TrtDataType type;     // 直接对应 C++ 的 TrtDataType
                                  // C++ 的 "void const*" 对应 C# 的 IntPtr
                                  // "const" 在这里只是一个承诺，P/Invoke 不关心它
        public IntPtr values;     // 指向数据块的内存地址
        public long count;        // C++ 的 int64_t 对应 C# 的 long (64位整数)
    }
    /// <summary>
    /// 一个用于安全管理和访问 Weights 结构体及其非托管内存的包装类。
    /// </summary>
    public sealed class Weights : IDisposable
    {
        // C++ Weights 结构体的 C# 副本，可以直接传递给 P/Invoke
        public TrtWeights NativeWeights { get; private set; }
        // 跟踪非托管内存的所有权，用于防止重复释放
        private bool isDisposed = false;
        // 指向我们自己分配的非托管内存块，用于后续释放
        private IntPtr ownedMemory = IntPtr.Zero;
        /// <summary>
        /// 获取每个元素的字节大小。
        /// </summary>
        public int ElementSize
        {
            get
            {
                return NativeWeights.type switch
                {
                    TrtDataType.kFLOAT => sizeof(float),
                    TrtDataType.kHALF => 2, // half 是 16 位
                    TrtDataType.kINT8 => sizeof(sbyte),
                    TrtDataType.kINT32 => sizeof(int),
                    TrtDataType.kBOOL => sizeof(bool),
                    _ => throw new NotSupportedException($"Unsupported data type: {NativeWeights.type}")
                };
            }
        }
        #region 构造函数
        /// <summary>
        /// 从现有的 Weights 结构体创建一个只读包装器。该包装器不管理内存。
        /// </summary>
        /// <param name="weights">一个已存在的 Weights 结构体。</param>
        public Weights(TrtWeights weights)
        {
            NativeWeights = weights;
            // ownedMemory 保持为 IntPtr.Zero，表示我们不拥有这块内存
        }
        /// <summary>
        /// 从 C# float 数组创建一个新的 Weights。该包装器会分配并管理非托管内存。
        /// </summary>
        /// <param name="data">要封装的浮点数据数组。</param>
        public Weights(float[] data)
        {
            if (data == null || data.Length == 0)
            {
                // Create an empty/null weights object
                NativeWeights = new TrtWeights { type = TrtDataType.kFLOAT, values = IntPtr.Zero, count = 0 };
                return;
            }
            AllocateMemoryAndCopy(data, TrtDataType.kFLOAT);
        }

        /// <summary>
        /// 从 C# int 数组创建一个新的 Weights。
        /// </summary>
        /// <param name="data">要封装的整数数据数组。</param>
        public Weights(int[] data)
        {
            if (data == null || data.Length == 0)
            {
                NativeWeights = new TrtWeights { type = TrtDataType.kINT32, values = IntPtr.Zero, count = 0 };
                return;
            }
            AllocateMemoryAndCopy(data, TrtDataType.kINT32);
        }

        //可以继续添加为其他数据类型（如sbyte[]）提供的构造函数...
        #endregion
        #region 私有辅助方法
        /// <summary>
        /// 分配非托管内存并从 C# 数组中复制数据。
        /// </summary>
        private void AllocateMemoryAndCopy<T>(T[] data, TrtDataType dataType) where T : struct
        {
            ownedMemory = Marshal.AllocHGlobal(data.Length * Marshal.SizeOf<T>());
            try
            {
                if (typeof(T) == typeof(int))
                {
                    Marshal.Copy((int[])(object)data, 0, ownedMemory, data.Length);
                }
                else if (typeof(T) == typeof(byte))
                {
                    Marshal.Copy((byte[])(object)data, 0, ownedMemory, data.Length);
                }
                else if (typeof(T) == typeof(float))
                {
                    Marshal.Copy((float[])(object)data, 0, ownedMemory, data.Length);
                }
                // ... 可以为 double, short, 等继续添加
                else
                {
                    throw new NotSupportedException($"Type {typeof(T).Name} is not supported for P/Invoke marshaling in this method.");
                }
                NativeWeights = new TrtWeights
                {
                    type = dataType,
                    values = ownedMemory,
                    count = data.Length
                };
            }
            catch
            {
                // 如果复制失败，确保释放已分配的内存
                Marshal.FreeHGlobal(ownedMemory);
                ownedMemory = IntPtr.Zero;
                throw;
            }
        }
        #endregion
        #region 数据访问方法
        /// <summary>
        /// 将非托管数据读取为 float 数组。
        /// </summary>
        /// <returns>包含数据副本的 float 数组。</returns>
        public float[] ReadAsFloats()
        {
            if (NativeWeights.type != TrtDataType.kFLOAT || NativeWeights.values == IntPtr.Zero || NativeWeights.count <= 0)
                return new float[0];
            float[] result = new float[NativeWeights.count];
            Marshal.Copy(NativeWeights.values, result, 0, result.Length);
            return result;
        }
        /// <summary>
        /// 将非托管数据读取为 int32 数组。
        /// </summary>
        /// <returns>包含数据副本的 int 数组。</returns>
        public int[] ReadAsInt32s()
        {
            if (NativeWeights.type != TrtDataType.kINT32 || NativeWeights.values == IntPtr.Zero || NativeWeights.count <= 0)
                return new int[0];
            int[] result = new int[NativeWeights.count];
            Marshal.Copy(NativeWeights.values, result, 0, result.Length);
            return result;
        }

        //可以继续添加其他类型的 Read 方法...
        #endregion
        #region IDisposable 实现
        /// <summary>
        /// 释放由 Weights 占用的所有非托管资源。
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this); // 防止终结器被调用
        }
        /// <summary>
        /// 释放资源的实际实现。
        /// </summary>
        /// <param name="disposing">如果由 Dispose() 调用，则为 true；如果由终结器调用，则为 false。</param>
        private void Dispose(bool disposing)
        {
            if (!isDisposed)
            {
                if (ownedMemory != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(ownedMemory);
                    ownedMemory = IntPtr.Zero;
                    // 可选：将结构体中的指针清零，以防止悬空指针
                    NativeWeights = new TrtWeights { type = NativeWeights.type, values = IntPtr.Zero, count = 0 };
                }
                isDisposed = true;
            }
        }
        /// <summary>
        /// 析构函数 (Finalizer)，作为安全网，以防忘记调用 Dispose()。
        /// </summary>
        ~Weights()
        {
            Dispose(false);
        }
        #endregion
    }

}
