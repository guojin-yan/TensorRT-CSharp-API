# split runtime package prototypes

This directory contains non-publishing prototypes for TensorRT 10 split-delivery runtime packages.

The current goal is to model how the large TensorRT 10 Windows runtime packages could be split before any broad public distribution:

- `Core` packages carry the bridge, CUDA runtime, and core TensorRT runtime libraries.
- `Extensions` packages carry builder resources, plugins, parser libraries, and other optional deployment assets.

These projects are prototypes only. They must not be published until:

- NVIDIA redistribution terms are reviewed.
- Package size targets are reviewed.
- A consumer validation flow proves that the selected combination works with the split package set.

Supporting scripts:

- `eng/Validate-SplitDeliveryPrototype.ps1`
- `eng/Collect-SplitRuntimeAssets.ps1`

