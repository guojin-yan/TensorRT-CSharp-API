using System.IO;
using System.Text;

namespace JYPPX.TensorRtSharp.Tools;

internal static class OnnxIdentityModel
{
    public static byte[] CreateDynamicBatchModel()
    {
        ProtoWriter model = new ProtoWriter();
        model.Int64(1, 8);
        model.String(2, "JYPPX.TensorRtSharp");
        model.Message(7, CreateGraph());
        model.Message(8, CreateOpsetImport(13));
        return model.ToArray();
    }

    private static byte[] CreateGraph()
    {
        ProtoWriter graph = new ProtoWriter();
        graph.Message(1, CreateIdentityNode());
        graph.String(2, "jyppx_dynamic_identity_graph");
        graph.Message(11, CreateValueInfo("input"));
        graph.Message(12, CreateValueInfo("output"));
        return graph.ToArray();
    }

    private static byte[] CreateIdentityNode()
    {
        ProtoWriter node = new ProtoWriter();
        node.String(1, "input");
        node.String(2, "output");
        node.String(3, "identity");
        node.String(4, "Identity");
        return node.ToArray();
    }

    private static byte[] CreateValueInfo(string name)
    {
        ProtoWriter valueInfo = new ProtoWriter();
        valueInfo.String(1, name);
        valueInfo.Message(2, CreateTensorFloatType());
        return valueInfo.ToArray();
    }

    private static byte[] CreateTensorFloatType()
    {
        ProtoWriter tensorType = new ProtoWriter();
        tensorType.Message(1, CreateTensorType());
        return tensorType.ToArray();
    }

    private static byte[] CreateTensorType()
    {
        ProtoWriter type = new ProtoWriter();
        type.UInt64(1, 1);
        type.Message(2, CreateShape());
        return type.ToArray();
    }

    private static byte[] CreateShape()
    {
        ProtoWriter shape = new ProtoWriter();
        shape.Message(1, CreateDimension("batch"));
        shape.Message(1, CreateDimension(4));
        return shape.ToArray();
    }

    private static byte[] CreateDimension(string parameterName)
    {
        ProtoWriter dimension = new ProtoWriter();
        dimension.String(2, parameterName);
        return dimension.ToArray();
    }

    private static byte[] CreateDimension(long value)
    {
        ProtoWriter dimension = new ProtoWriter();
        dimension.Int64(1, value);
        return dimension.ToArray();
    }

    private static byte[] CreateOpsetImport(long version)
    {
        ProtoWriter opset = new ProtoWriter();
        opset.Int64(2, version);
        return opset.ToArray();
    }
}

internal sealed class ProtoWriter
{
    private readonly MemoryStream _stream = new MemoryStream();

    public void Int64(int fieldNumber, long value)
    {
        WriteTag(fieldNumber, 0);
        WriteVarint(unchecked((ulong)value));
    }

    public void UInt64(int fieldNumber, ulong value)
    {
        WriteTag(fieldNumber, 0);
        WriteVarint(value);
    }

    public void String(int fieldNumber, string value)
    {
        Bytes(fieldNumber, Encoding.UTF8.GetBytes(value));
    }

    public void Message(int fieldNumber, byte[] value)
    {
        Bytes(fieldNumber, value);
    }

    public byte[] ToArray()
    {
        return _stream.ToArray();
    }

    private void Bytes(int fieldNumber, byte[] value)
    {
        WriteTag(fieldNumber, 2);
        WriteVarint((ulong)value.Length);
        _stream.Write(value, 0, value.Length);
    }

    private void WriteTag(int fieldNumber, int wireType)
    {
        WriteVarint((ulong)((fieldNumber << 3) | wireType));
    }

    private void WriteVarint(ulong value)
    {
        while (value >= 0x80)
        {
            _stream.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        _stream.WriteByte((byte)value);
    }
}
