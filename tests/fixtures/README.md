# Test Fixtures

These files are copied into isolated temporary projects by repository quality scripts. They are not samples,
applications, publishable projects, or proof that a public package is available.

| Directory | Purpose | Publication boundary |
| --- | --- | --- |
| `package-consumers` | Active local managed/bridge package isolation checks | Never published |
| `mnist-onnx-runtime-reference` | Independent ONNX Runtime reference generation | Never published |
| `legacy-package-consumers` | Reproduction of pre-release Classification/YoloVision evidence | Retired; historical validation only |

Fixture projects use `.csproj.template` so they cannot be restored or packed accidentally by a solution-wide project
scan. Validation scripts materialize them outside the repository and must keep publication disabled.
