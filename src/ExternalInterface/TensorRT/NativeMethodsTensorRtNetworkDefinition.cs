using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        // Network Definition Functions
        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtNetworkDefinition_free(IntPtr network);

        // --- Network Input and Output ---
        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addInput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addInput(
            IntPtr network,
            [MarshalAs(StringUnmanagedTypeNotWindows)]  string name,
            TrtDataType type,
            ref Dims dimensions,
            out IntPtr input);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_markOutput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_markOutput(
            IntPtr network,
            IntPtr tensor);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_unmarkOutput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_unmarkOutput(
            IntPtr network,
            IntPtr tensor);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_markOutputForShapes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_markOutputForShapes(
            IntPtr network,
            IntPtr tensor,
            ref int wasMarked);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_unmarkOutputForShapes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_unmarkOutputForShapes(
            IntPtr network,
            IntPtr tensor,
            ref int wasUnmarked);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_markDebug",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_markDebug(
            IntPtr network,
            IntPtr tensor,
            ref int wasMarked);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_unmarkDebug",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_unmarkDebug(
            IntPtr network,
            IntPtr tensor,
            ref int wasUnmarked);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_isDebugTensor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_isDebugTensor(
            IntPtr network,
            IntPtr tensor,
            ref int isDebug);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_removeTensor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_removeTensor(
            IntPtr network,
            IntPtr tensor);

        // --- Network Query ---
        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getNbLayers",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getNbLayers(
            IntPtr network,
            out int nbLayers);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getLayer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getLayer(
            IntPtr network,
            int index,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getNbInputs",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getNbInputs(
            IntPtr network,
            out int nbInputs);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getInput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getInput(
            IntPtr network,
            int index,
            out IntPtr input);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getNbOutputs",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getNbOutputs(
            IntPtr network,
            out int nbOutputs);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getOutput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getOutput(
            IntPtr network,
            int index,
            out IntPtr output);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_setName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_setName(
            IntPtr network,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getName(
            IntPtr network,
            out IntPtr name);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getFlags(
            IntPtr network,
            out uint flags);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getFlag",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getFlag(
            IntPtr network,
            TrtNetworkDefinitionCreationFlag flag,
            out int isSet);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_getBuilder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_getBuilder(
            IntPtr network,
            out IntPtr builder);

        // --- Layer Factory Functions ---
        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addActivation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addActivation(
            IntPtr network,
            IntPtr input,
            TrtActivationType type,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addLRN",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addLRN(
            IntPtr network,
            IntPtr input,
            long window,
            float alpha,
            float beta,
            float k,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addScale",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addScale(
            IntPtr network,
            IntPtr input,
            TrtScaleMode mode,
            ref TrtWeights shift,
            ref TrtWeights scale,
            ref TrtWeights power,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addSoftMax",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addSoftMax(
            IntPtr network,
            IntPtr input,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addConcatenation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addConcatenation(
            IntPtr network,
            IntPtr[] inputs,
            int nbInputs,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addElementWise",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addElementWise(
            IntPtr network,
            IntPtr input1,
            IntPtr input2,
            TrtElementWiseOperation op,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addUnary",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addUnary(
            IntPtr network,
            IntPtr input,
            TrtUnaryOperation operation,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addShuffle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addShuffle(
            IntPtr network,
            IntPtr input,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addOneHot",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addOneHot(
            IntPtr network,
            IntPtr indices,
            IntPtr values,
            IntPtr depth,
            int axis,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addReduce",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addReduce(
            IntPtr network,
            IntPtr input,
            TrtReduceOperation operation,
            uint reduceAxes,
            int keepDimensions,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addTopK",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addTopK(
            IntPtr network,
            IntPtr input,
            TrtTopKOperation op,
            int k,
            uint reduceAxes,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addGather",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addGather(
            IntPtr network,
            IntPtr data,
            IntPtr indices,
            int axis,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addGatherV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addGatherV2(
            IntPtr network,
            IntPtr data,
            IntPtr indices,
            TrtGatherMode mode,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addRaggedSoftMax",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addRaggedSoftMax(
            IntPtr network,
            IntPtr input,
            IntPtr bounds,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addMatrixMultiply",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addMatrixMultiply(
            IntPtr network,
            IntPtr input0,
            TrtMatrixOperation op0,
            IntPtr input1,
            TrtMatrixOperation op1,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addNonZero",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addNonZero(
            IntPtr network,
            IntPtr input,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addConstant",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addConstant(
            IntPtr network,
            ref Dims dimensions,
            ref TrtWeights weights,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addIdentity",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addIdentity(
            IntPtr network,
            IntPtr input,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addCast",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addCast(
            IntPtr network,
            IntPtr input,
            TrtDataType toType,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addSlice",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addSlice(
            IntPtr network,
            IntPtr input,
            ref Dims start,
            ref Dims size,
            ref Dims stride,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addShape",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addShape(
            IntPtr network,
            IntPtr input,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addParametricReLU",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addParametricReLU(
            IntPtr network,
            IntPtr input,
            IntPtr slope,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addConvolutionNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addConvolutionNd(
            IntPtr network,
            IntPtr input,
            int nbOutputMaps,
            ref Dims kernelSize,
            ref TrtWeights kernelWeights,
            ref TrtWeights biasWeights,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addPoolingNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addPoolingNd(
            IntPtr network,
            IntPtr input,
            TrtPoolingType type,
            ref Dims windowSize,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addDeconvolutionNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addDeconvolutionNd(
            IntPtr network,
            IntPtr input,
            int nbOutputMaps,
            ref Dims kernelSize,
            ref TrtWeights kernelWeights,
            ref TrtWeights biasWeights,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addScaleNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addScaleNd(
            IntPtr network,
            IntPtr input,
            TrtScaleMode mode,
            ref TrtWeights shift,
            ref TrtWeights scale,
            ref TrtWeights power,
            int channelAxis,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addResize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addResize(
            IntPtr network,
            IntPtr input,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addLoop",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addLoop(
            IntPtr network,
            out IntPtr loop);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addIfConditional",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addIfConditional(
            IntPtr network,
            out IntPtr conditional);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addSelect",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addSelect(
            IntPtr network,
            IntPtr condition,
            IntPtr thenInput,
            IntPtr elseInput,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addAssertion",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addAssertion(
            IntPtr network,
            IntPtr condition,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string message,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addFill",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addFill(
            IntPtr network,
            ref Dims dimensions,
            TrtFillOperation op,
            TrtDataType outputType,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addPaddingNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addPaddingNd(
            IntPtr network,
            IntPtr input,
            ref Dims prePadding,
            ref Dims postPadding,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addDequantize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addDequantize(
            IntPtr network,
            IntPtr input,
            IntPtr scale,
            TrtDataType outputType,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addScatter",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addScatter(
            IntPtr network,
            IntPtr data,
            IntPtr indices,
            IntPtr updates,
            TrtScatterMode mode,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addQuantize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addQuantize(
            IntPtr network,
            IntPtr input,
            IntPtr scale,
            TrtDataType outputType,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addEinsum",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addEinsum(
            IntPtr network,
            IntPtr[] inputs,
            int nbInputs,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string equation,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addGridSample",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addGridSample(
            IntPtr network,
            IntPtr input,
            IntPtr grid,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addNMS",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addNMS(
            IntPtr network,
            IntPtr boxes,
            IntPtr scores,
            IntPtr maxOutputBoxesPerClass,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addReverseSequence",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addReverseSequence(
            IntPtr network,
            IntPtr input,
            IntPtr sequenceLens,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addNormalization",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addNormalization(
            IntPtr network,
            IntPtr input,
            IntPtr scale,
            IntPtr bias,
            uint axesMask,
            out IntPtr layer);

        // --- Plugin and other advanced functions ---
        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_addPluginV3",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_addPluginV3(
            IntPtr network,
            IntPtr[] inputs,
            int nbInputs,
            IntPtr[] shapeInputs,
            int nbShapeInputs,
            ref IntPtr plugin,
            out IntPtr layer);

        [DllImport(dllExtern, EntryPoint = "trtNetworkDefinition_setWeightsName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtNetworkDefinition_setWeightsName(
            IntPtr network,
            ref TrtWeights weights,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name,
            out int wasSet);

    }
}
