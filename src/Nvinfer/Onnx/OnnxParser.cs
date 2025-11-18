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

using System;
using System.Runtime.InteropServices;

/// <summary>
/// OnnxParser类，用于解析ONNX模型并将其转换为TensorRT网络定义。
/// OnnxParser class for parsing ONNX models and converting them to TensorRT network definitions.
/// </summary>
/// <remarks>
/// 此类继承自DisposableTrtObject，提供了解析ONNX模型的API，支持从内存中解析或从文件解析。
/// This class inherits from DisposableTrtObject and provides APIs for parsing ONNX models, 
/// supporting parsing from memory or from files.
/// </remarks>
public class OnnxParser : DisposableTrtObject
{
    /// <summary>
    /// 构造函数，使用给定的网络定义初始化ONNX解析器。
    /// Constructor, initializes the ONNX parser with a given network definition.
    /// </summary>
    /// <param name="networkDefinition">
    /// TensorRT网络定义对象，解析器将向其中填充网络的层和张量。
    /// TensorRT network definition object, which the parser will populate with the network's layers and tensors.
    /// </param>
    /// <exception cref="Exception">
    /// 当创建解析器失败时抛出异常。
    /// Thrown when the parser creation fails.
    /// </exception>
    public OnnxParser(NetworkDefinition networkDefinition)
    {
        InitHandleException.handler(
            NativeMethods.trtOnnxParser_createOnnxParser(networkDefinition.TrtPtr, out ptr));
    }

    /// <summary>
    /// 释放当前解析器持有的所有资源。此方法为 Dispose 的显式别名。
    /// Releases all resources held by the current parser. This method is an explicit alias for Dispose.
    /// </summary>
    public void Release()
    {
        Dispose();
    }

    /// <inheritdoc />
    /// <summary>
    /// 释放所有非托管资源。此方法由 Dispose 模式调用，不应直接调用。
    /// Releases all unmanaged resources. This method is called by the Dispose pattern and should not be called directly.
    /// </summary>
    protected override void DisposeUnmanaged()
    {
        //if (ptr != IntPtr.Zero && IsEnabledDispose)
        //    NativeMethods.trtOnnxParser_free(ptr); // 假设正确的函数是 trtOnnxParser_free
        base.DisposeUnmanaged();
    }

    /// <summary>
    /// 从内存中的序列化ONNX模型解析网络。
    /// Parse the network from a serialized ONNX model in memory.
    /// </summary>
    /// <param name="serialized_onnx_model">
    /// 指向序列化ONNX模型数据的指针。
    /// Pointer to the serialized ONNX model data.
    /// </param>
    /// <param name="serialized_onnx_model_size">
    /// 序列化ONNX模型的大小（以字节为单位）。
    /// Size of the serialized ONNX model in bytes.
    /// </param>
    /// <param name="model_path">
    /// 模型文件路径，用于解析过程中可能需要的相对路径引用（例如，用于加载外部权重文件）。
    /// Model file path, used for relative path references that may be needed during parsing (e.g., for loading external weight files).
    /// </param>
    /// <returns>
    /// 如果解析成功返回true，否则返回false。
    /// Returns true if parsing is successful, otherwise false.
    /// </returns>
    public bool parse(IntPtr serialized_onnx_model, long serialized_onnx_model_size, string model_path)
    {
        TrtHandleException.handler(NativeMethods.trtONNXParser_parse(ptr,
            serialized_onnx_model,
            serialized_onnx_model_size,
            model_path,
            out int flag));
        return flag != 0;
    }

    /// <summary>
    /// 从ONNX模型文件解析网络。
    /// Parse the network from an ONNX model file.
    /// </summary>
    /// <param name="onnxModelFile">
    /// ONNX模型文件的路径。
    /// Path to the ONNX model file.
    /// </param>
    /// <param name="verbosity">
    /// 解析过程中的详细级别，用于控制日志输出量。
    /// Verbosity level during parsing, to control the amount of log output.
    /// </param>
    /// <returns>
    /// 如果解析成功返回true，否则返回false。
    /// Returns true if parsing is successful, otherwise false.
    /// </returns>
    public bool parseFromFile(string onnxModelFile, int verbosity = 0)
    {
        TrtHandleException.handler(NativeMethods.trtONNXParser_parseFromFile(ptr,
            onnxModelFile,
            verbosity,
            out int flag));
        return flag != 0;
    }

    /// <summary>
    /// 检查解析器是否支持指定的ONNX操作符。
    /// Checks if the parser supports the specified ONNX operator.
    /// </summary>
    /// <param name="opName">要检查的操作符名称。/ Name of the operator to check.</param>
    /// <returns>如果支持该操作符，则为true；否则为false。/ True if the operator is supported, otherwise false.</returns>
    public bool supportsOperator(string opName)
    {
        TrtHandleException.handler(NativeMethods.trtParser_supportsOperator(ptr,
            opName,
            out int supported));
        return supported != 0;
    }

    /// <summary>
    /// 获取在解析过程中产生的错误数量。
    /// Gets the number of errors that occurred during parsing.
    /// </summary>
    /// <returns>错误的数量。/ The number of errors.</returns>
    public int getNbErrors()
    {
        TrtHandleException.handler(NativeMethods.trtParser_getNbErrors(ptr,
            out int nbErrors));
        return nbErrors;
    }

    /// <summary>
    /// 获取指定索引处的错误详情。
    /// Gets the error details at a given index.
    /// </summary>
    /// <param name="index">错误的索引，范围从 0 到 NbErrors - 1。/ Index of the error, ranging from 0 to NbErrors - 1.</param>
    /// <returns>一个包含错误信息的 ParserError 对象。/ A ParserError object containing the error information.</returns>
    public ParserError getError(int index)
    {
        TrtHandleException.handler(NativeMethods.trtParser_getError(ptr,
            index,
            out IntPtr error));
        return new ParserError(error);
    }

    /// <summary>
    /// 清除所有解析错误。
    /// Clears all parsing errors.
    /// </summary>
    public void clearErrors()
    {
        TrtHandleException.handler(NativeMethods.trtParser_clearErrors(ptr));
    }

    /// <summary>
    /// 获取解析模型时使用的所有VisionWorks（VC）插件库名称。
    /// Gets the names of all VisionWorks (VC) plugin libraries used when parsing the model.
    /// </summary>
    /// <returns>一个包含插件库名称的字符串数组。/ A string array containing the names of the plugin libraries.</returns>
    public string[] getUsedVCPluginLibraries()
    {
        TrtHandleException.handler(NativeMethods.trtParser_getUsedVCPluginLibraries(ptr,
            out long nbPluginLibs,
            out IntPtr pluginLibrariesPtr));
        string[] pluginLibraries = new string[nbPluginLibs];
        for (long i = 0; i < nbPluginLibs; i++)
        {
            IntPtr currentPtr = Marshal.ReadIntPtr(pluginLibrariesPtr, (int)(i * IntPtr.Size));
            pluginLibraries[i] = Marshal.PtrToStringAnsi(currentPtr);
        }
        return pluginLibraries;
    }

    /// <summary>
    /// 设置解析器的标志位组合，用于控制解析行为。
    /// Sets the combination of parser flags, used to control parsing behavior.
    /// </summary>
    /// <param name="onnxParserFlags">要设置的标志位组合。/ The flag combination to set.</param>
    public void setFlags(uint onnxParserFlags)
    {
        TrtHandleException.handler(NativeMethods.trtParser_setFlags(ptr, onnxParserFlags));
    }

    /// <summary>
    /// 获取当前的解析器标志位组合。
    /// Gets the current combination of parser flags.
    /// </summary>
    /// <returns>当前的标志位组合。/ The current flag combination.</returns>
    public uint getFlags()
    {
        TrtHandleException.handler(NativeMethods.trtParser_getFlags(ptr, out uint onnxParserFlags));
        return onnxParserFlags;
    }

    /// <summary>
    /// 清除指定的解析器标志位。
    /// Clears the specified parser flag.
    /// </summary>
    /// <param name="onnxParserFlag">要清除的标志位。/ The flag to clear.</param>
    public void clearFlag(TrtOnnxParserFlag onnxParserFlag)
    {
        TrtHandleException.handler(NativeMethods.trtParser_clearFlag(ptr, onnxParserFlag));
    }

    /// <summary>
    /// 设置指定的解析器标志位。
    /// Sets the specified parser flag.
    /// </summary>
    /// <param name="onnxParserFlag">要设置的标志位。/ The flag to set.</param>
    public void setFlag(TrtOnnxParserFlag onnxParserFlag)
    {
        TrtHandleException.handler(NativeMethods.trtParser_setFlag(ptr, onnxParserFlag));
    }

    /// <summary>
    /// 检查是否设置了指定的解析器标志位。
    /// Checks if a specific parser flag is set.
    /// </summary>
    /// <param name="onnxParserFlag">要检查的标志位。/ The flag to check.</param>
    /// <returns>如果设置了该标志位，则为true；否则为false。/ True if the flag is set, otherwise false.</returns>
    public bool getFlag(TrtOnnxParserFlag onnxParserFlag)
    {
        TrtHandleException.handler(NativeMethods.trtParser_getFlag(ptr, onnxParserFlag, out int isSet));
        return isSet != 0;
    }

    /// <summary>
    /// 根据层名称和索引获取该层的输出张量。
    /// Gets the output tensor of a layer based on its name and output index.
    /// </summary>
    /// <param name="name">层的名称。/ Name of the layer.</param>
    /// <param name="i">输出张量的索引。/ The index of the output tensor.</param>
    /// <returns>对应的 Tensor 对象。/ The corresponding Tensor object.</returns>
    public Tensor getLayerOutputTensor(string name, long i)
    {
        TrtHandleException.handler(NativeMethods.trtParser_getLayerOutputTensor(ptr,
            name,
            i,
            out IntPtr tensorPtr));
        return new Tensor(tensorPtr);
    }

    /// <summary>
    /// 检查给定的序列化ONNX模型是否受支持（更高版本的检查）。
    /// Checks if the given serialized ONNX model is supported (a newer version check).
    /// </summary>
    /// <param name="serializedOnnxModel">指向序列化ONNX模型数据的指针。/ Pointer to the serialized ONNX model data.</param>
    /// <param name="serializedOnnxModelSize">模型数据大小（字节）。/ Size of the model data in bytes.</param>
    /// <param name="modelPath">模型路径，用于解析外部引用。/ Model path, used for parsing external references.</param>
    /// <returns>如果模型受支持，则为true；否则为false。/ True if the model is supported, otherwise false.</returns>
    public bool supportsModelV2(IntPtr serializedOnnxModel, ulong serializedOnnxModelSize, string modelPath)
    {
        TrtHandleException.handler(NativeMethods.trtParser_supportsModelV2(ptr,
            serializedOnnxModel,
            serializedOnnxModelSize,
            modelPath,
            out int success));
        return success != 0;
    }

    /// <summary>
    /// 获取模型中子图的数量。
    /// Gets the number of subgraphs in the model.
    /// </summary>
    /// <returns>子图的数量。/ The number of subgraphs.</returns>
    public long getNbSubgraphs()
    {
        TrtHandleException.handler(NativeMethods.trtParser_getNbSubgraphs(ptr,
            out long nbSubgraphs));
        return nbSubgraphs;
    }

    /// <summary>
    /// 检查指定索引的子图是否受支持。
    /// Checks if a subgraph at the specified index is supported.
    /// </summary>
    /// <param name="index">子图的索引。/ Index of the subgraph.</param>
    /// <returns>如果子图受支持，则为true；否则为false。/ True if the subgraph is supported, otherwise false.</returns>
    public bool isSubgraphSupported(long index)
    {
        TrtHandleException.handler(NativeMethods.trtParser_isSubgraphSupported(ptr,
            index,
            out int isSupported));
        return isSupported != 0;
    }

    /// <summary>
    /// 获取指定子图中包含的节点ID列表。
    /// Gets the list of node IDs contained in the specified subgraph.
    /// </summary>
    /// <param name="index">子图的索引。/ Index of the subgraph.</param>
    /// <returns>一个包含节点ID的long数组。/ A long array containing the node IDs.</returns>
    public long[] getSubgraphNodes(long index)
    {
        TrtHandleException.handler(NativeMethods.trtParser_getSubgraphNodes(ptr,
            index,
            out IntPtr subgraphNodesPtr,
            out long subgraphLength));
        long[] subgraphNodes = new long[subgraphLength];
        Marshal.Copy(subgraphNodesPtr, subgraphNodes, 0, (int)subgraphLength);
        return subgraphNodes;
    }
}


}
