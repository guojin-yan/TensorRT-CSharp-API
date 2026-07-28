using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Owns a reusable native NVRTC program boundary without exposing the native handle. 拥有可复用 native NVRTC program 边界且不暴露 native handle。</summary>
public sealed class CudaRtcProgram : IDisposable
{
    private readonly SafeCudaRtcProgramHandle _handle;
    private bool _disposed;

    /// <summary>Creates an owner from an immutable copied source snapshot. 从不可变复制型 source 快照创建 owner。</summary>
    public CudaRtcProgram(CudaRtcProgramSource source)
    {
        Source = source ?? throw new ArgumentNullException(nameof(source));
        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.CreateRtcProgram(source);
    }

    /// <summary>Gets the immutable managed source snapshot. 获取不可变托管 source 快照。</summary>
    public CudaRtcProgramSource Source { get; }

    /// <summary>Compiles with a copied option snapshot; compiler failures are returned with their full log. 使用复制型 option 快照编译；compiler failure 会连同完整日志返回。</summary>
    public CudaRtcCompilationResult Compile(CudaRtcCompileOptions? options = null)
    {
        ThrowIfDisposed();
        CudaRtcCompileOptions effectiveOptions = options ?? CudaRtcCompileOptions.Default;
        int nativeResult = NativeCudaApi.CompileRtcProgram(_handle, effectiveOptions.Options);
        var resultCode = (CudaRtcResultCode)nativeResult;
        string log = NativeCudaApi.GetRtcProgramLog(_handle);
        CudaRtcCapability capability = CudaRtcCompiler.GetCapability();

        var artifacts = new List<CudaRtcArtifact>();
        var loweredNames = new List<CudaRtcLoweredName>();
        if (resultCode == CudaRtcResultCode.Success)
        {
            string sourceHash = CudaRtcHash.Text(Source.Source);
            var headerCanonical = new List<string>();
            foreach (CudaRtcHeader header in Source.Headers)
            {
                headerCanonical.Add(header.IncludeName);
                headerCanonical.Add(header.Source);
            }
            string headersHash = CudaRtcHash.Sequence(headerCanonical);
            string optionsHash = CudaRtcHash.Sequence(effectiveOptions.Options);

            AddArtifactIfAvailable(artifacts, CudaRtcArtifactKind.Ptx, sourceHash, headersHash, optionsHash, capability.Version, effectiveOptions.TargetArchitecture);
            AddArtifactIfAvailable(artifacts, CudaRtcArtifactKind.Cubin, sourceHash, headersHash, optionsHash, capability.Version, effectiveOptions.TargetArchitecture);
            AddArtifactIfAvailable(artifacts, CudaRtcArtifactKind.LtoIr, sourceHash, headersHash, optionsHash, capability.Version, effectiveOptions.TargetArchitecture);

            for (int index = 0; index < Source.NameExpressions.Count; index++)
            {
                loweredNames.Add(new CudaRtcLoweredName(
                    Source.NameExpressions[index],
                    NativeCudaApi.GetRtcLoweredName(_handle, checked((uint)index))));
            }
        }

        return new CudaRtcCompilationResult(resultCode, log, capability.Version, artifacts, loweredNames);
    }

    /// <summary>Releases the native program owner. 释放 native program owner。</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _handle.Dispose();
        _disposed = true;
    }

    private void AddArtifactIfAvailable(
        ICollection<CudaRtcArtifact> artifacts,
        CudaRtcArtifactKind kind,
        string sourceHash,
        string headersHash,
        string optionsHash,
        string compilerVersion,
        string targetArchitecture)
    {
        byte[]? bytes = NativeCudaApi.TryCopyRtcArtifact(_handle, kind);
        if (bytes != null)
        {
            artifacts.Add(new CudaRtcArtifact(
                kind,
                bytes,
                sourceHash,
                headersHash,
                optionsHash,
                compilerVersion,
                targetArchitecture));
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed || _handle.IsClosed || _handle.IsInvalid)
        {
            throw new ObjectDisposedException(nameof(CudaRtcProgram));
        }
    }
}
