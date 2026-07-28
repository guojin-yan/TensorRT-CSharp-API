using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Owns one asynchronous CUDA Driver kernel launch and its participating owners. 拥有一次异步 CUDA Driver kernel launch 及其参与的所有 owner。</summary>
public sealed class CudaDriverKernelLaunch : IDisposable
{
    private const int MaximumArgumentCount = 256;
    private const int MaximumScalarDataSize = 64 * 1024;

    private readonly SafeCudaDriverKernelLaunchHandle _handle;
    private readonly SafeCudaHandleLease _ownerLease;
    private readonly object _lifecycleGate = new object();
    private bool _disposed;

    private CudaDriverKernelLaunch(
        SafeCudaDriverKernelLaunchHandle handle,
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

    /// <summary>Gets the copied function name. 获取复制后的 function name。</summary>
    public string KernelName { get; }

    /// <summary>Gets the copied launch configuration. 获取复制后的 launch configuration。</summary>
    public CudaKernelLaunchConfiguration Configuration { get; }

    /// <summary>Gets the number of packed arguments. 获取打包后的参数数量。</summary>
    public int ArgumentCount { get; }

    /// <summary>Gets whether the Driver completion event has finished. 获取 Driver completion event 是否完成。</summary>
    public bool IsCompleted
    {
        get
        {
            lock (_lifecycleGate)
            {
                ThrowIfDisposed();
                return NativeCudaApi.QueryDriverKernelLaunch(_handle);
            }
        }
    }

    /// <summary>Waits for Driver kernel completion and surfaces asynchronous errors. 等待 Driver kernel 完成并报告异步错误。</summary>
    public void Synchronize()
    {
        lock (_lifecycleGate)
        {
            ThrowIfDisposed();
            NativeCudaApi.SynchronizeDriverKernelLaunch(_handle);
        }
    }

    /// <summary>Synchronizes pending work, destroys the Driver launch owner, and releases all leases. 同步未完成工作、销毁 Driver launch owner 并释放全部租约。</summary>
    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the Driver completion owner and its SafeHandle leases. 释放 Driver completion owner 及其 SafeHandle 租约。</summary>
    ~CudaDriverKernelLaunch()
    {
        DisposeCore();
    }

    internal static CudaDriverKernelLaunch Create(
        SafeCudaDriverModuleHandle module,
        string kernelName,
        CudaKernelLaunchConfiguration configuration,
        CudaStream stream,
        CudaKernelArgument[] arguments)
    {
        if (arguments.Length > MaximumArgumentCount)
        {
            throw new ArgumentOutOfRangeException(nameof(arguments), "CUDA Driver typed launch accepts at most 256 arguments.");
        }

        configuration.Validate();
        var handles = new List<SafeHandle>(2 + arguments.Length) { module, stream.Handle };
        int[] memoryLeaseIndices = new int[arguments.Length];
        for (int index = 0; index < arguments.Length; ++index)
        {
            CudaKernelArgument argument = arguments[index] ?? throw new ArgumentException("CUDA Driver kernel arguments must not contain null values.", nameof(arguments));
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
                    while (scalarData.Count % alignment != 0) scalarData.Add(0);
                    int offset = scalarData.Count;
                    scalarData.AddRange(bytes);
                    if (scalarData.Count > MaximumScalarDataSize)
                    {
                        throw new ArgumentOutOfRangeException(nameof(arguments), "CUDA Driver typed launch scalar payload exceeds 64 KiB.");
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

            SafeCudaDriverKernelLaunchHandle launch = NativeCudaApi.LaunchDriverModule(
                module,
                kernelName,
                configuration,
                nativeArguments,
                scalarData.ToArray(),
                stream.Handle);
            return new CudaDriverKernelLaunch(launch, lease, kernelName, configuration, arguments.Length);
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
            if (_disposed) return;
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
        if (_disposed) throw new ObjectDisposedException(nameof(CudaDriverKernelLaunch));
    }
}
