using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class OnnxEngineBuildDiagnostics
{
    private static OnnxEngineBuildReportBoundary CreateReportBoundary(OnnxEngineBuildResult result)
    {
        return new OnnxEngineBuildReportBoundary(
            isRuntimeProof: false,
            isBuildOnly: result.BuildEvidenceOnly,
            forbiddenSubstituteReason: "TensorRtExec reports are diagnostic/build artifacts. They do not replace real-model-runtime, package-consumer-runtime, post-publish verification, or owner release-close evidence.",
            copiedDiagnosticsBoundary: "ONNX Parser and ParserRefitter copied diagnostics are troubleshooting and release-gate surface evidence only. They do not prove engine execution, model correctness, package-consumer-runtime, post-publish verification, or release close readiness.",
            parserDiagnosticsEvidenceKind: "copied-parser-diagnostics",
            parserRefitterDiagnosticsEvidenceKind: "copied-parser-refitter-diagnostics",
            canPromoteCopiedDiagnosticsToRuntimeProof: false,
            parserDiagnosticsOwnerAction: "Use copied parser/refitter diagnostics to repair ONNX export, shape/profile, or plugin plans; then collect real-model-runtime evidence with real inputs, output JSON, logs, hashes, host metadata, and owner review.",
            forbiddenSubstitutes: new[]
            {
                "build-only",
                "dry-run",
                "template",
                "local feed",
                "ProjectReference",
                "direct `.nupkg`",
                "TensorRtExec report",
                "YoloVision matrix",
                "OnnxToEngine report",
                "readonly diagnostics",
                "capability-probe-only",
                "ONNX Parser diagnostic snapshot",
                "ONNX ParserRefitter diagnostic snapshot",
                "copied-parser-diagnostics",
                "copied-parser-refitter-diagnostics"
            });
    }
}
