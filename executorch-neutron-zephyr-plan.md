# Plan: ExecuTorch with the Neutron NPU on Zephyr (i.MX RT700)

Status as of 2026-10-15. Builds on [executorch-zephyr-study.md](executorch-zephyr-study.md).
Target: `mimxrt700_evk/mimxrt798s/cm33_cpu0`, ExecuTorch **v1.4.0**, eIQ Neutron SDK **3.2.3**.
Nothing in this plan has been built or run yet.

## Goal

Export a PyTorch model on the host, run it on the RT700 under Zephyr with the parts the Neutron
NPU supports delegated to it, and finally call it from an mpipe pipeline, as `jpeg_npu` does
with TFLM.

## Decisions to take first

| Decision | Choice | Reason |
| :--- | :--- | :--- |
| ExecuTorch version | Tag `v1.4.0` | Version the Zephyr docs pin; reproducible |
| Neutron version | **3.2.3 everywhere**: pip `eiq-neutron-sdk==3.2.3` for the export, SDK 3.2.3 libraries and header on target | The driver rejects microcode from another version. pip has 3.2.3; ExecuTorch v1.4.0 was tested with **3.1.3**, so 3.2.3 must be validated (fallback: 3.1.3 for all three) |
| Driver source | SDK libraries, overriding `hal_nxp`'s 3.0.0 blobs | Same approach as `jpeg_npu` |
| Reference model | CifarNet (`examples/nxp`, `-m cifar10`) then MobileNetV2 | The two models NXP lists as supported; CifarNet is the one already run with TFLM on this board |
| Tensor layout | Export with `--use_channels_last_dim_order` | Avoids the CPU NCHW↔NHWC transposes `NeutronBackend::execute()` otherwise performs |

## Step-by-step plan

### Phase 1: Workspace and host tools

| # | Step | Done when |
| :--- | :--- | :--- |
| 1.1 | Add `zephyr/submanifests/executorch.yaml` (`revision: v1.4.0`, `submodules: true`, path `modules/lib/executorch`) | File in place |
| 1.2 | `west config manifest.project-filter -- +tflite-micro,+executorch,+cmsis-nn` then `west update executorch` | `west list executorch` shows v1.4.0 |
| 1.3 | Create a venv; in `modules/lib/executorch`: `git submodule update --init --recursive`, `./install_executorch.sh` | `python -c "import executorch"` works |
| 1.4 | `pip install eiq-neutron-sdk==3.2.3 --index-url https://eiq.nxp.com/repository/` (overrides the 3.1.3 pin in `backends/nxp/requirements-eiq.txt`) | `python -c "from eiq_neutron_sdk import neutron_converter"` works |

### Phase 2: CPU-only baseline on RT700

Proves that ExecuTorch itself runs on the board before any NPU work.

| # | Step | Done when |
| :--- | :--- | :--- |
| 2.1 | Export CifarNet without delegation: `python -m examples.nxp.aot_neutron_compile -m cifar10 --quantize` | `.pte` produced; its operator list contains only CPU kernels |
| 2.2 | Copy `zephyr/samples/hello-executorch` to the workspace as a new sample; add `boards/mimxrt700_evk_mimxrt798s_cm33_cpu0.conf` (FPU, heap, method/temp pool sizes, `CONFIG_ETHOS_U` off) | Builds for RT700 |
| 2.3 | Build with `-DET_PTE_FILE_PATH=<pte>`, flash, compare the output with the same `.pte` run on the host (ExecuTorch pybindings) | Same top class; scores within int8 rounding |

### Phase 3: NPU export on the host

| # | Step | Done when |
| :--- | :--- | :--- |
| 3.1 | `python -m examples.nxp.aot_neutron_compile -m cifar10 --quantize --delegate --target imxrt700 --use_channels_last_dim_order --remove-quant-io-ops` | `cifar10_nxp_delegate.pte` produced with 3.2.3 |
| 3.2 | Check delegation: the conversion log / ExecuTorch inspector shows the NPU partition, few CPU ops left | Most of the graph in one `NeutronBackend` delegate |
| 3.3 | Optional: run it on the host Neutron simulator (`eiq_nsys`, `examples/nxp/run.sh`) | Same result as 2.3 |

### Phase 4: Wire the Neutron backend into the Zephyr module

Patch `modules/lib/executorch/zephyr/CMakeLists.txt` (kept as a local patch, upstreamed in
Phase 8):

| # | Step | Detail |
| :--- | :--- | :--- |
| 4.1 | Backend selection | `if(CONFIG_NXP_NEUTRON) set(EXECUTORCH_BUILD_NXP_NEUTRON ON) endif()` next to the Ethos-U case |
| 4.2 | Compile flags | Add `executorch_delegate_neutron` to `EXECUTORCH_TARGETS_TO_FIX` so it builds with Zephyr's arch flags |
| 4.3 | Driver linkage | Link `executorch_delegate_neutron` to the `neutron_driver` and `neutron_firmware` imported targets from `hal_nxp`, re-pointed at the SDK 3.2.3 `.a` files |
| 4.4 | Registration | In the app, link the delegate with `executorch_target_link_options_shared_lib()` (whole-archive), else the static `register_backend()` object is dropped by the linker |

### Phase 5: Fix the Neutron backend for the 3.x driver

| # | Step | Detail |
| :--- | :--- | :--- |
| 5.1 | Header | Compile `NeutronBackend.cpp` against the SDK 3.2.3 `NeutronDriver.h`/`NeutronErrors.h`, not the copies in `backends/nxp/runtime/`: the bundled `NeutronModelConfig` lacks `subgraphName` |
| 5.2 | No driver heap | Build with `NO_HEAP_USAGE=1` so the model context comes from ExecuTorch's runtime allocator (`neutronGetModelContextSize()`), not `malloc` |
| 5.3 | External memory | Leave `EXTERNAL_MEM` undefined (model in SRAM); revisit only for large models |
| 5.4 | Logging types | Check the `%ld` of `NeutronError` (an `int32_t`) builds without warnings on Zephyr |

### Phase 6: Application for the RT700

| # | Step | Detail |
| :--- | :--- | :--- |
| 6.1 | Kconfig | `CONFIG_EXECUTORCH=y`, `CONFIG_NXP_NEUTRON=y` (also powers the NPU in `board.c`), `CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n` (selective build), `CONFIG_FPU=y` |
| 6.2 | Memory placement | Put the `.pte` copy, the method pool and the temp pool in **SRAM1** (`Z_GENERIC_SECTION("SRAM1")`), inside the 5.5 MiB the NPU can reach. The NPU reads microcode and weights from the `.pte` and writes scratch from the temp pool |
| 6.3 | Cache | Keep `CONFIG_DCACHE=n` at first (as `jpeg_dec`); later, non-cacheable section or flush/invalidate around `execute()` |
| 6.4 | Model location | Leave `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=n` so the runner copies the model from flash to SRAM |
| 6.5 | Pool sizes | Start large (e.g. 256 KiB each), then shrink from the runner's reported usage |

### Phase 7: Validate on hardware

| # | Check | Pass criterion |
| :--- | :--- | :--- |
| 7.1 | `neutronModelPrepare()` succeeds | No `ERROR_CATEGORY_DRIVER_UCODE` (version) or `BAD_HANDLE` |
| 7.2 | Output correctness | Matches Phase 2 (CPU) and the host run of the delegated `.pte` |
| 7.3 | Latency | `execute()` time vs the CPU baseline and vs TFLM + Neutron on the same CifarNet (`tflm_neutron` sample) |
| 7.4 | Memory | Flash and SRAM usage vs the TFLM build |
| 7.5 | MobileNetV2 | Repeat 3.1 to 7.3 with `-m mobilenetv2` |

### Phase 8: mpipe integration and upstreaming

| # | Step | Detail |
| :--- | :--- | :--- |
| 8.1 | Backend choice in `jpeg_npu` | Add an ExecuTorch variant of `src/inference.cpp` with the same `inference_init()`/`inference_run()` API, chosen by Kconfig; the pipeline and preprocessing stay unchanged |
| 8.2 | Upstream to ExecuTorch | Zephyr module Neutron hook (Phase 4), 3.x header (Phase 5), an RT700 board conf in a Zephyr sample |
| 8.3 | Upstream to Zephyr / hal_nxp | Neutron blobs at the matching version, so the SDK override is no longer needed |

## Risks

| Risk | Effect | Mitigation |
| :--- | :--- | :--- |
| ExecuTorch NXP Python was tested with 3.1.3, not 3.2.3 | Export may fail or emit an unexpected payload | Fall back to 3.1.3 for export **and** target libraries (SDK 3.1.3 package needed) |
| Struct mismatch in the bundled header | Driver reads a garbage `subgraphName` | Phase 5.1 |
| Upstream support is MCUXpresso-only | Zephyr-specific issues are ours to solve | Keep the patches small, upstream them |
| Memory: `.pte` + pools + video buffers in SRAM1 | Overflow with larger models | Size from Phase 7.4; video pool and model share the 2 MiB SRAM1 |
| No NPU interrupt in Zephyr | `neutronRunBlocking()` holds the calling thread | Run inference in its own thread (as `jpeg_npu` does) |

## How it will work when done

### Build time

```
 HOST                                                           ZEPHYR BUILD
 cifarnet.py (torch.nn.Module)
   │ torch.export
   │ NeutronQuantizer (int8)
   │ NeutronPartitioner ── supported subgraph ─▶ TFLite (in memory)
   │                                              │ eiq_neutron_sdk 3.2.3
   │                                              ▼ neutron_converter --target imxrt700
   │                        microcode + weights (+ scratch size) = delegate payload
   ▼
 model.pte  ──── -DET_PTE_FILE_PATH ────▶ gen_oplist.py: CPU ops still in the graph
                                          → selective kernel build (portable / cortex_m)
                                          pte_to_header.py → model_pte.h (flash)
                                          ExecuTorch module CMake:
                                            EXECUTORCH_BUILD_NXP_NEUTRON (CONFIG_NXP_NEUTRON)
                                            executorch_delegate_neutron
                                            + libNeutronDriver.a / libNeutronFirmware.a (3.2.3)
                                          → zephyr.elf
```

### Run time

```
 boot: board.c powers the NPU (CONFIG_NXP_NEUTRON)
 app thread
   runtime_init()
   memcpy(model_pte, flash → SRAM1)
   Program::load(BufferDataLoader(model_pte))
   program->load_method("forward", memory_manager{method pool, planned memory, temp pool} in SRAM1)
     └─ NeutronBackend::init()        payload check, neutronModelPrepare(microcode, weights)
 per frame (mpipe app_sink → inference thread)
   preprocess frame → method.set_input(tensor)   int8 NHWC, written in SRAM1
   method->execute()
     ├─ CPU ops (if any)  → portable / CMSIS-NN kernels on the Cortex-M33
     └─ delegate          → NeutronBackend::execute()
                              scratch = temp pool (SRAM1)
                              neutronResume() → neutronRunBlocking() → neutronSuspend()
                                                     │
                                                     ▼
                                     Neutron NPU + firmware: run microcode,
                                     read input/weights, write output in SRAM1
   method->get_outputs() → scores → log / downstream element
```

Compared with today's `jpeg_npu` (TFLM): the pipeline, the preprocessing, the SRAM1 placement and
the Neutron driver/firmware are the same. What changes is the authoring tool (PyTorch instead of
TensorFlow), the file format (`.pte` instead of `.tflite`), the runtime (ExecuTorch `Method`
instead of TFLM `MicroInterpreter`) and the glue (`NeutronBackend` delegate instead of the
`NeutronGraph` custom op).
