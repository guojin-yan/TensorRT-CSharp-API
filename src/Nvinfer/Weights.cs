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

    /// <summary>
    /// 表示 TensorRT 权重的非托管布局结构。<br/>
    /// Represents the unmanaged layout structure for TensorRT weights.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TrtWeights
    {
        /// <summary>
        /// 权重数据的数据类型。<br/>
        /// The data type of the weights.
        /// </summary>
        public TrtDataType type;

        /// <summary>
        /// 指向权重数据块的内存地址。<br/>
        /// Pointer to the memory address of the weights data block.
        /// </summary>
        public IntPtr values;

        /// <summary>
        /// 权重元素的数量。<br/>
        /// The count of weight elements.
        /// </summary>
        public long count;
    }

    /// <summary>
    /// 一个用于安全管理和访问 <see cref="TrtWeights"/> 结构体及其非托管内存的包装类。<br/>
    /// A wrapper class for safely managing and accessing the <see cref="TrtWeights"/> struct and its unmanaged memory.
    /// </summary>
    public sealed class Weights : IDisposable
    {
        /// <summary>
        /// 获取底层的、可直接用于互操作的原生 <see cref="TrtWeights"/> 结构体。<br/>
        /// Gets the underlying native <see cref="TrtWeights"/> struct, which can be used directly for interop.
        /// </summary>
        public TrtWeights NativeWeights { get; private set; }

        /// <summary>
        /// 跟踪非托管内存的所有权，用于防止重复释放。<br/>
        /// Tracks the ownership of the unmanaged memory to prevent double-freeing.
        /// </summary>
        private bool isDisposed = false;

        /// <summary>
        /// 指向我们自己分配的非托管内存块，用于后续释放。<br/>
        /// Pointer to the unmanaged memory block we allocated, used for later disposal.
        /// </summary>
        private IntPtr ownedMemory = IntPtr.Zero;

        /// <summary>
        /// 获取每个元素的字节大小。<br/>
        /// Gets the size of each element in bytes.
        /// </summary>
        /// <exception cref="NotSupportedException">
        /// 如果权重数据类型不受支持，则抛出此异常。<br/>
        /// Thrown if the weight data type is not supported.
        /// </exception>
        public int ElementSize
        {
            get
            {
                return NativeWeights.type switch
                {
                    TrtDataType.kFLOAT => sizeof(float),
                    TrtDataType.kHALF => 2, // half is 16-bit
                    TrtDataType.kINT8 => sizeof(sbyte),
                    TrtDataType.kINT32 => sizeof(int),
                    TrtDataType.kBOOL => sizeof(bool),
                    _ => throw new NotSupportedException($"Unsupported data type: {NativeWeights.type}")
                };
            }
        }

        #region 构造函数
        /// <summary>
        /// 从现有的 <see cref="TrtWeights"/> 结构体创建一个只读包装器。该包装器不管理内存。<br/>
        /// Creates a read-only wrapper from an existing <see cref="TrtWeights"/> struct. This wrapper does not own the memory.
        /// </summary>
        /// <param name="weights">一个已存在的 <see cref="TrtWeights"/> 结构体。<br/>An existing <see cref="TrtWeights"/> struct.</param>
        public Weights(TrtWeights weights)
        {
            NativeWeights = weights;
            // ownedMemory remains IntPtr.Zero, indicating we do not own this memory.
        }

        /// <summary>
        /// 从 C# <c>float</c> 数组创建一个新的 <see cref="Weights"/>。该包装器会分配并管理非托管内存。<br/>
        /// Creates a new <see cref="Weights"/> from a C# <c>float</c> array. The wrapper allocates and manages the unmanaged memory.
        /// </summary>
        /// <param name="data">要封装的浮点数据数组。<br/>The array of float data to wrap.</param>
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
        /// 从 C# <c>int</c> 数组创建一个新的 <see cref="Weights"/>。<br/>
        /// Creates a new <see cref="Weights"/> from a C# <c>int</c> array.
        /// </summary>
        /// <param name="data">要封装的整数数据数组。<br/>The array of integer data to wrap.</param>
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
        /// 分配非托管内存并从 C# 数组中复制数据。<br/>
        /// Allocates unmanaged memory and copies data from a C# array.
        /// </summary>
        /// <typeparam name="T">数组元素的类型。<br/>The type of the array elements.</typeparam>
        /// <param name="data">源数据数组。<br/>The source data array.</param>
        /// <param name="dataType">对应的 TensorRT 数据类型。<br/>The corresponding TensorRT data type.</param>
        /// <exception cref="NotSupportedException">如果指定的类型 <typeparamref name="T"/> 不受支持，则抛出此异常。<br/>Thrown if the specified type <typeparamref name="T"/> is not supported.</exception>
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
                // If copying fails, ensure the allocated memory is released.
                Marshal.FreeHGlobal(ownedMemory);
                ownedMemory = IntPtr.Zero;
                throw;
            }
        }
        #endregion

        #region 数据访问方法
        /// <summary>
        /// 将非托管数据读取为 <c>float</c> 数组。<br/>
        /// Reads the unmanaged data as a <c>float</c> array.
        /// </summary>
        /// <returns>包含数据副本的 <c>float</c> 数组。<br/>A <c>float</c> array containing a copy of the data.</returns>
        public float[] ReadAsFloats()
        {
            if (NativeWeights.type != TrtDataType.kFLOAT || NativeWeights.values == IntPtr.Zero || NativeWeights.count <= 0)
                return new float[0];
            float[] result = new float[NativeWeights.count];
            Marshal.Copy(NativeWeights.values, result, 0, result.Length);
            return result;
        }

        /// <summary>
        /// 将非托管数据读取为 <c>int32</c> 数组。<br/>
        /// Reads the unmanaged data as a <c>int32</c> array.
        /// </summary>
        /// <returns>包含数据副本的 <c>int</c> 数组。<br/>An <c>int</c> array containing a copy of the data.</returns>
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
        /// 释放由 <see cref="Weights"/> 占用的所有非托管资源。<br/>
        /// Releases all unmanaged resources used by the <see cref="Weights"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this); // Prevent the finalizer from being called.
        }

        /// <summary>
        /// 释放资源的实际实现。<br/>
        /// The actual implementation for releasing resources.
        /// </summary>
        /// <param name="disposing">
        /// 如果由 <see cref="Dispose()"/> 调用，则为 true；如果由终结器调用，则为 false。<br/>
        /// <c>true</c> if called from <see cref="Dispose()"/>; <c>false</c> if called from a finalizer.
        /// </param>
        private void Dispose(bool disposing)
        {
            if (!isDisposed)
            {
                if (ownedMemory != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(ownedMemory);
                    ownedMemory = IntPtr.Zero;
                    // Optional: Zero out the pointer in the struct to prevent dangling pointers.
                    NativeWeights = new TrtWeights { type = NativeWeights.type, values = IntPtr.Zero, count = 0 };
                }
                isDisposed = true;
            }
        }

        /// <summary>
        /// 析构函数 (Finalizer)，作为安全网，以防忘记调用 <see cref="Dispose()"/>。<br/>
        /// Finalizer, acting as a safety net in case <see cref="Dispose()"/> is not called.
        /// </summary>
        ~Weights()
        {
            Dispose(false);
        }
        #endregion
    }


}
