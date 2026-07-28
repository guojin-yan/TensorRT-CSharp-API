using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns one asynchronous named-kernel launch and retains every participating CUDA owner until disposal.
/// 拥有一次异步 named-kernel launch，并在释放前租用所有参与的 CUDA owner。
/// </summary>
public sealed class CudaKernelLaunch : IDisposable
{
    private const int MaximumArgumentCount = 256;
    private const int MaximumScalarDataSize = 64 * 1024;

    private readonly SafeCudaKernelLaunchHandle _handle;
    private readonly SafeCudaHandleLease _ownerLease;
    private readonly object _lifecycleGate = new object();
    private bool _disposed;

    private CudaKernelLaunch(
        SafeCudaKernelLaunchHandle handle,
        SafeCudaHandleLease ownerLease,
        string kernelName,
        CudaKernelLaunchConfiguration configuration,
        int argumentCount)
    {
        _handle = handle;
        _ownerLease = ownerLease;
        KernelName = kernelName;
        Configuration = configuration;
        ArgumentCount = argumentCount;
    }

    /// <summary>Gets the copied kernel name. 获取复制后的 kernel 名称。</summary>
    public string KernelName { get; }

    /// <summary>Gets the copied launch configuration. 获取复制后的 launch 配置。</summary>
    public CudaKernelLaunchConfiguration Configuration { get; }

    /// <summary>Gets the number of packed kernel arguments. 获取打包后的 kernel 参数数量。</summary>
    public int ArgumentCount { get; }

    /// <summary>Gets whether the completion event has finished without exposing the native event. 获取 completion event 是否完成，不暴露 native event。</summary>
    public bool IsCompleted
    {
        get
        {
            lock (_lifecycleGate)
            {
                ThrowIfDisposed();
                return NativeCudaApi.QueryKernelLaunch(_handle);
            }
        }
    }

    /// <summary>Waits for kernel completion and surfaces asynchronous CUDA errors. 等待 kernel 完成并报告异步 CUDA 错误。</summary>
    public void Synchronize()
    {
        lock (_lifecycleGate)
        {
            ThrowIfDisposed();
            NativeCudaApi.SynchronizeKernelLaunch(_handle);
        }
    }

    /// <summary>Synchronizes pending work, destroys the completion owner, and releases all SafeHandle leases. 同步未完成工作、销毁 completion owner，并释放全部 SafeHandle 租约。</summary>
    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the completion owner and its SafeHandle leases. 释放 completion owner 及其 SafeHandle 租约。</summary>
    ~CudaKernelLaunch()
    {
        DisposeCore();
    }

    internal static CudaKernelLaunch Create(
        SafeCudaKernelLibraryHandle library,
        string kernelName,
        CudaKernelLaunchConfiguration configuration,
        CudaStream stream,
        CudaKernelArgument[] arguments)
    {
        if (arguments.Length > MaximumArgumentCount)
        {
            throw new ArgumentOutOfRangeException(nameof(arguments), "CUDA typed launch accepts at most 256 arguments.");
        }

        configuration.Validate();
        var handles = new List<SafeHandle>(2 + arguments.Length) { library, stream.Handle };
        int[] memoryLeaseIndices = new int[arguments.Length];
        for (int index = 0; index < arguments.Length; ++index)
        {
            CudaKernelArgument argument = arguments[index] ?? throw new ArgumentException("CUDA kernel arguments must not contain null values.", nameof(arguments));
            memoryLeaseIndices[index] = -1;
            if (argument.Kind == CudaKernelArgumentKind.DeviceMemory)
            {
                memoryLeaseIndices[index] = handles.Count;
                handles.Add(argument.Memory.Handle);
            }
        }

        SafeCudaHandleLease lease = SafeCudaHandleLease.Create(handles);
        try
        {
            var scalarData = new List<byte>();
            var nativeArguments = new NativeCudaKernelArgumentDescriptor[arguments.Length];
            for (int index = 0; index < arguments.Length; ++index)
            {
                CudaKernelArgument argument = arguments[index];
                if (argument.Kind == CudaKernelArgumentKind.Scalar)
                {
                    byte[] bytes = argument.ScalarBytes;
                    int alignment = Math.Min(bytes.Length, sizeof(long));
                    while (scalarData.Count % alignment != 0)
                    {
                        scalarData.Add(0);
                    }
                    int offset = scalarData.Count;
                    scalarData.AddRange(bytes);
                    if (scalarData.Count > MaximumScalarDataSize)
                    {
                        throw new ArgumentOutOfRangeException(nameof(arguments), "CUDA typed launch scalar payload exceeds 64 KiB.");
                    }
                    nativeArguments[index] = new NativeCudaKernelArgumentDescriptor
                    {
                        Kind = (int)CudaKernelArgumentKind.Scalar,
                        ScalarOffset = new UIntPtr((uint)offset),
                        ScalarSize = new UIntPtr((uint)bytes.Length)
                    };
                }
                else
                {
                    nativeArguments[index] = new NativeCudaKernelArgumentDescriptor
                    {
                        Kind = (int)CudaKernelArgumentKind.DeviceMemory,
                        Memory = lease.GetHandle(memoryLeaseIndices[index]),
                        MemoryOffset = new UIntPtr((uint)argument.MemoryOffset)
                    };
                }
            }

            SafeCudaKernelLaunchHandle launch = NativeCudaApi.LaunchKernelLibrary(
                lease.GetHandle(0),
                kernelName,
                configuration,
                nativeArguments,
                scalarData.ToArray(),
                lease.GetHandle(1));
            return new CudaKernelLaunch(launch, lease, kernelName, configuration, arguments.Length);
        }
        catch
        {
            lease.Dispose();
            throw;
        }
    }

    private void DisposeCore()
    {
        lock (_lifecycleGate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            try
            {
                _handle.Dispose();
            }
            finally
            {
                _ownerLease.Dispose();
            }
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaKernelLaunch));
        }
    }
}
