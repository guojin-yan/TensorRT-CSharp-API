using System;
using System.IO;
using System.Text;

namespace OnnxBuildAndRunSample;

internal static class SyntheticIdentityOnnxModel
{
    public static string WriteToTemporaryDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-tensorrt-samples", "onnx-build-and-run");
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "identity-1x4.onnx");
        File.WriteAllBytes(path, CreateModel());
        return path;
    }

    private static byte[] CreateModel()
    {
        ProtoWriter model = new ProtoWriter();
        model.WriteVarint(1, 9);
        model.WriteString(2, "TensorRtSharp4.0");
        model.WriteString(3, "4.0");
        model.WriteMessage(7, WriteGraph);
        model.WriteMessage(8, static opset => opset.WriteVarint(2, 13));
        return model.ToArray();
    }

    private static void WriteGraph(ProtoWriter graph)
    {
        graph.WriteMessage(1, static node =>
        {
            node.WriteString(1, "input");
            node.WriteString(2, "output");
            node.WriteString(3, "identity");
            node.WriteString(4, "Identity");
        });
        graph.WriteString(2, "synthetic_identity_1x4");
        graph.WriteMessage(11, input => WriteValueInfo(input, "input"));
        graph.WriteMessage(12, output => WriteValueInfo(output, "output"));
    }

    private static void WriteValueInfo(ProtoWriter valueInfo, string name)
    {
        valueInfo.WriteString(1, name);
        valueInfo.WriteMessage(2, type => type.WriteMessage(1, tensorType =>
        {
            tensorType.WriteVarint(1, 1);
            tensorType.WriteMessage(2, shape =>
            {
                shape.WriteMessage(1, static dimension => dimension.WriteVarint(1, 1));
                shape.WriteMessage(1, static dimension => dimension.WriteVarint(1, 4));
            });
        }));
    }

    private sealed class ProtoWriter
    {
        private readonly MemoryStream _stream = new MemoryStream();

        public void WriteVarint(int fieldNumber, ulong value)
        {
            WriteTag(fieldNumber, 0);
            WriteRawVarint(value);
        }

        public void WriteString(int fieldNumber, string value)
        {
            WriteBytes(fieldNumber, Encoding.UTF8.GetBytes(value));
        }

        public void WriteMessage(int fieldNumber, Action<ProtoWriter> write)
        {
            ProtoWriter child = new ProtoWriter();
            write(child);
            WriteBytes(fieldNumber, child.ToArray());
        }

        public byte[] ToArray()
        {
            return _stream.ToArray();
        }

        private void WriteBytes(int fieldNumber, byte[] value)
        {
            WriteTag(fieldNumber, 2);
            WriteRawVarint((ulong)value.Length);
            _stream.Write(value, 0, value.Length);
        }

        private void WriteTag(int fieldNumber, int wireType)
        {
            WriteRawVarint((ulong)((fieldNumber << 3) | wireType));
        }

        private void WriteRawVarint(ulong value)
        {
            while (value >= 0x80)
            {
                _stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            _stream.WriteByte((byte)value);
        }
    }
}
