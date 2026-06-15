# Local GitHub Actions Checks

GitHub Actions workflows are still the source of truth for release publication, but part of the workflow graph can be checked locally with [`act`](https://github.com/nektos/act).

## What local `act` is good for

Use local `act` runs for fast, non-publishing checks:

- Parse workflow files and validate job graphs.
- Dry-run `workflow_dispatch` inputs and expressions.
- Check hosted Linux orchestration jobs such as `release-bundle.yml` and the `runtime-linux.yml` `prepare` job.
- Reproduce small script failures without creating extra GitHub Actions run records.

Example dry-runs:

```powershell
act workflow_dispatch -W .github/workflows/runtime-linux.yml -j prepare -n -P ubuntu-latest=catthehacker/ubuntu:act-latest --pull=false

act workflow_dispatch -W .github/workflows/release-bundle.yml -j orchestrate -n -P ubuntu-latest=catthehacker/ubuntu:act-latest --pull=false
```

## What local `act` should not be trusted for

Do not treat local `act` as release evidence for this repository:

- Windows hosted jobs are not faithfully reproduced by a Linux container image.
- Self-hosted Windows runtime jobs still require this machine's installed CUDA, cuDNN, TensorRT, Visual Studio, CMake, signing, and package-source configuration.
- Linux runtime publishing still requires a real self-hosted Linux x64 runner with matching NVIDIA roots.
- `act` dry-runs do not upload artifacts, publish packages, deploy Pages, or prove package restore from GitHub Releases/GitHub Packages.

For release evidence, dispatch through `gh workflow run` and inspect the GitHub-hosted or self-hosted runner logs.

## Current local setup note

On Windows, `winget install --id nektos.act -e` installs `act.exe`, but the current PowerShell session may not refresh `PATH` until a new shell is opened. If needed, call the installed executable directly from the WinGet package directory or open a new terminal.

Docker Desktop must be running before non-dry-run `act` jobs can start containers.

