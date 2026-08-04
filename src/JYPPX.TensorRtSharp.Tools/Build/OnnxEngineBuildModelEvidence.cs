using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildModelEvidence
{
    public OnnxEngineBuildModelEvidence(
        string modelSource,
        string modelSha256,
        string modelLicense,
        string inputAssetName,
        string inputAssetSha256)
    {
        ModelSource = modelSource ?? string.Empty;
        ModelSha256 = modelSha256 ?? string.Empty;
        ModelLicense = modelLicense ?? string.Empty;
        InputAssetName = inputAssetName ?? string.Empty;
        InputAssetSha256 = inputAssetSha256 ?? string.Empty;
    }

    public string ModelSource { get; }

    public string ModelSha256 { get; }

    public string ModelLicense { get; }

    public string InputAssetName { get; }

    public string InputAssetSha256 { get; }
}
