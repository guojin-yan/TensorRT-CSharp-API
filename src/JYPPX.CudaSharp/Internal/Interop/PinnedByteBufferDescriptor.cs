namespace JYPPX.CudaSharp.Internal.Interop;

internal sealed class PinnedByteBufferDescriptor
{
    public PinnedByteBufferDescriptor(string operationName, string bufferParameterName, string nativeEntryPoint, string transferDirection)
    {
        OperationName = operationName;
        BufferParameterName = bufferParameterName;
        NativeEntryPoint = nativeEntryPoint;
        TransferDirection = transferDirection;
    }

    public string OperationName { get; }
    public string BufferParameterName { get; }
    public string NativeEntryPoint { get; }
    public string TransferDirection { get; }
}
