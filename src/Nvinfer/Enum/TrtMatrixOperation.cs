using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 枚举了在矩阵乘法层中，对输入张量在乘法之前可以执行的预操作。<br/>
    /// Enumerates the operations that may be performed on a tensor by IMatrixMultiplyLayer before multiplication.
    /// </summary>
    public enum TrtMatrixOperation : int
    {
        /// <summary>
        /// 不对张量应用特殊操作。<br/>
        /// No special operation is applied to the tensor.
        /// </summary>
        /// <remarks>
        /// 如果张量 `x` 有两个维度，则将其视为矩阵；如果 `x` 有超过两个维度，则将其视为矩阵集合，其中最后两个维度是该集合中每个矩阵的维度。`x` 必须至少有两个维度。<br/>
        /// Treat x as a matrix if it has two dimensions, or as a collection of matrices if x has more than two dimensions, where the last two dimensions are the matrix dimensions. x must have at least two dimensions.
        /// </remarks>
        kNONE = 0,

        /// <summary>
        /// 对张量应用转置操作，但除此之外与 kNONE 相同。<br/>
        /// Like kNONE, but transpose the matrix dimensions.
        /// </summary>
        /// <remarks>
        /// 这与 kNONE 规则类似，但会对张量的矩阵维度（通常是最后两个维度）执行转置。<br/>
        /// This is similar to the kNONE rule, but a transpose is performed on the matrix dimensions (typically the last two dimensions) of the tensor.
        /// </remarks>
        kTRANSPOSE = 1,

        /// <summary>
        /// 将张量视为向量或向量集合。<br/>
        /// Treat the tensor as a vector or a collection of vectors.
        /// </summary>
        /// <remarks>
        /// 如果张量 `x` 有一个维度，则将其视为单个向量；如果 `x` 有多个维度，则将其视为向量集合。`x` 必须至少有一个维度。<br/>
        /// If x has one dimension, it is treated as a single vector. If x has more than one dimension, it is treated as a collection of vectors. x must have at least one dimension.
        /// 
        /// <para>
        /// 此模式的行为取决于该张量是矩阵乘法的第一个还是第二个输入：
        /// <br/>
        /// The behavior of this mode depends on whether the tensor is the first or second input of the matrix multiplication:
        /// </para>
        /// <list type="bullet">
        /// <item>
        /// <description>
        /// <b>作为第一个输入</b>：维度为 [M, K] 的张量在使用 <c>MatrixOperation::kVECTOR</c> 时，等价于一个维度为 [M, 1, K] 并使用 <c>MatrixOperation::kNONE</c> 的张量。它被视作 M 个长度为 K 的行向量。
        /// <br/>
        /// <b>As the first input</b>: A tensor with dimensions [M,K] used with <c>MatrixOperation::kVECTOR</c> is equivalent to a tensor with dimensions [M, 1, K] with <c>MatrixOperation::kNONE</c>, i.e., it is treated as M row vectors of length K.
        /// </description>
        /// </item>
        /// <item>
        /// <description>
        /// <b>作为第二个输入</b>：维度为 [M, K] 的张量在使用 <c>MatrixOperation::kVECTOR</c> 时，等价于一个维度为 [M, K, 1] 并使用 <c>MatrixOperation::kNONE</c> 的张量。它被视作 M 个长度为 K 的列向量。
        /// <br/>
        /// <b>As the second input</b>: A tensor with dimensions [M,K] used with <c>MatrixOperation::kVECTOR</c> is equivalent to a tensor with dimensions [M, K, 1] with <c>MatrixOperation::kNONE</c>, i.e., it is treated as M column vectors of length K.
        /// </description>
        /// </item>
        /// </list>
        /// <para>
        /// 在这两种情况下，都可以通过使用 <c>MatrixOperation::kTRANSPOSE</c> 来调换向量集合的维度。
        /// <br/>
        /// In both cases, <c>MatrixOperation::kTRANSPOSE</c> can be used to swap the dimensions of the vector collections.
        /// </para>
        /// </remarks>
        kVECTOR = 2,
    };

}
