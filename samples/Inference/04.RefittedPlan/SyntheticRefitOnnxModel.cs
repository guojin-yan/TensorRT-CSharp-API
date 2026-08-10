using System;
using System.Buffers.Binary;
using System.IO;
using System.Text;

namespace RefittedPlanSample;

internal sealed class SyntheticRefitOnnxModel
{
    private SyntheticRefitOnnxModel(string baselinePath, string refitPath)
    {
        BaselinePath = baselinePath;
        RefitPath = refitPath;
    }

    public string BaselinePath { get; }

    public string RefitPath { get; }

    public static SyntheticRefitOnnxModel WriteToTemporaryDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-tensorrt-samples", "refitted-plan");
        Directory.CreateDirectory(directory);
        string baselinePath = Path.Combine(directory, "scale-1x4-baseline.onnx");
        string refitPath = Path.Combine(directory, "scale-1x4-refit.onnx");
        File.WriteAllBytes(baselinePath, CreateModel(new[] { 1.0f, 1.0f, 1.0f, 1.0f }));
        File.WriteAllBytes(refitPath, CreateModel(new[] { 2.0f, 2.0f, 2.0f, 2.0f }));
        return new SyntheticRefitOnnxModel(baselinePath, refitPath);
    }

    private static byte[] CreateModel(float[] scaleValues)
    {
        ProtoWriter model = new ProtoWriter();
        model.WriteVarint(1, 9);
        model.WriteString(2, "TensorRtSharp4.0");
        model.WriteString(3, "4.0");
        model.WriteMessage(7, graph => WriteGraph(graph, scaleValues));
        model.WriteMessage(8, static opset => opset.WriteVarint(2, 13));
        return model.ToArray();
    }

    private static void WriteGraph(ProtoWriter graph, float[] scaleValues)
    {
        graph.WriteMessage(1, static node =>
        {
            node.WriteString(1, "input");
            node.WriteString(1, "scale");
            node.WriteString(2, "output");
            node.WriteString(3, "scale_mul");
            node.WriteString(4, "Mul");
        });
        graph.WriteString(2, "synthetic_refit_scale_1x4");
        graph.WriteMessage(5, initializer => WriteInitializer(initializer, scaleValues));
        graph.WriteMessage(11, input => WriteValueInfo(input, "input"));
        graph.WriteMessage(12, output => WriteValueInfo(output, "output"));
    }

    private static void WriteInitializer(ProtoWriter tensor, float[] values)
    {
        tensor.WriteVarint(1, 1);
        tensor.WriteVarint(1, 4);
        tensor.WriteVarint(2, 1);
        tensor.WriteString(8, "scale");

        byte[] rawData = new byte[checked(values.Length * sizeof(float))];
        for (int index = 0; index < values.Length; index++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(rawData.AsSpan(index * sizeof(float), sizeof(float)), values[index]);
        }
        tensor.WriteBytes(9, rawData);
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

        public void WriteBytes(int fieldNumber, byte[] value)
        {
            WriteTag(fieldNumber, 2);
            WriteRawVarint((ulong)value.Length);
            _stream.Write(value, 0, value.Length);
        }

        public byte[] ToArray()
        {
            return _stream.ToArray();
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
