using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal sealed class TrtexecLikeIoFormatSpec
{
    public TrtexecLikeIoFormatSpec(string dataTypeToken, TensorRtDataType dataType, string formatText, TensorRtTensorFormats formats)
    {
        DataTypeToken = dataTypeToken;
        DataType = dataType;
        FormatText = formatText;
        Formats = formats;
    }

    public string DataTypeToken { get; }

    public TensorRtDataType DataType { get; }

    public string FormatText { get; }

    public TensorRtTensorFormats Formats { get; }

    public override string ToString() => DataTypeToken + ":" + FormatText;
}
