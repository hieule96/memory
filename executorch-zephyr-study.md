# ExecuTorch on Zephyr: integration study (4W)

Status as of 2026-10-15. Read from the ExecuTorch sources at tag **v1.4.0** (the revision the Zephyr
documentation pins), mainly `zephyr/`, `backends/nxp/`, `backends/arm/runtime/`,
`runtime/platform/default/zephyr.cpp` and `examples/nxp/`, plus the Zephyr page
"ExecuTorch external module". Nothing here was built or run.

Each topic is answered with **What / Why / Where / When**.

## Summary

| Topic | Finding |
| :--- | :--- |
| 1. Manifest | ExecuTorch is an **external** west project with its own `zephyr/module.yml`; it is not in Zephyr's `west.yml` |
| 2. Build flow | The module's CMake runs ExecuTorch's own CMake under Zephyr's toolchain and flags; the model's operator list drives a selective kernel build |
| 3. Model deployment | PyTorch model exported on the host to a `.pte` file, embedded in the image as a C array, loaded from memory at run time |
| 4. Runtime | ExecuTorch's own C++ runtime. **No PyTorch and no TensorFlow on the device**; PyTorch is used only on the host |
| 5. NPU | Only **Arm Ethos-U** is wired into the Zephyr module. A Neutron backend exists in ExecuTorch but is **not integrated with Zephyr**, and has version and header mismatches to solve first |

```
 HOST (Python, PyTorch)                                  TARGET (Zephyr, C++)
 torch.nn.Module
   │ torch.export + quantizer (Arm or NXP)
   │ partitioner: NPU subgraphs → delegate payload       ┌─────────────────────────────┐
   │   Ethos-U: Vela command stream                      │ app (executor runner)       │
   │   Neutron: TFLite → neutron_converter microcode     │  Program::load(.pte)        │
   ▼                                                     │  Method::execute()          │
 model.pte ──pte_to_header.py──▶ model_pte.h ──────────▶ │   CPU ops: portable /       │
                                                         │            cortex_m (CMSIS) │
                                                         │   delegate: EthosUBackend   │
                                                         │            (NeutronBackend: │
                                                         │             not wired)      │
                                                         │  PAL: k_malloc, printk...   │
                                                         └─────────────────────────────┘
```

---

## 1. Manifest integration

**What.** ExecuTorch is pulled in as a west project from `github.com/pytorch/executorch` at
`modules/lib/executorch`. It ships `zephyr/module.yml` (`name: executorch`, `build.cmake: zephyr`,
`build.kconfig: zephyr/Kconfig`), so Zephyr discovers it as a module once west has fetched it.
It is **not** in Zephyr's `west.yml` or `submanifests/optional.yaml` (unlike `tflite-micro`).

**Why.** ExecuTorch is a large, fast-moving PyTorch project with Python tooling and many
third-party submodules. Keeping it external lets it pin its own Zephyr version and release
cadence, rather than Zephyr pinning it.

**Where.**

| File | Content |
| :--- | :--- |
| `executorch/zephyr/module.yml` | Module declaration |
| `executorch/zephyr/executorch.yaml` | Ready-made submanifest, `revision: main` |
| `zephyr/submanifests/executorch.yaml` (you create it) | Zephyr docs example uses `revision: v1.4.0`, `submodules: true` |

```yaml
manifest:
  projects:
    - name: executorch
      url: https://github.com/pytorch/executorch
      revision: v1.4.0
      path: modules/lib/executorch
      submodules: true
```

**When.** Set it up once per workspace, before building:

```
west config manifest.project-filter -- +executorch,+cmsis-nn   # plus +hal_ethos_u for Ethos-U
west update executorch
cd modules/lib/executorch && git submodule update --init --recursive && ./install_executorch.sh
```

`install_executorch.sh` installs the Python side (PyTorch and the AOT tools) in the active venv.
ExecuTorch's README targets Zephyr **v4.4.0**. Pin a tag, not `main`, to get reproducible builds.

---

## 2. Build flow

**What.** `CONFIG_EXECUTORCH=y` (it selects `REQUIRES_FULL_LIBCPP`) makes the module's
`zephyr/CMakeLists.txt`:

1. Load ExecuTorch's `tools/cmake/preset/zephyr.cmake`: turns off everything an MCU cannot use
   (pybind, XNNPACK, threadpool, data-loader extension, devtools, ...).
2. Pick backends from Zephyr Kconfig:
   - `CONFIG_CPU_CORTEX_M` → `EXECUTORCH_BUILD_CORTEX_M` (CMSIS-NN kernels, from the workspace's
     `modules/lib/cmsis-nn`)
   - `CONFIG_CPU_CORTEX_A` → XNNPACK
   - `CONFIG_ETHOS_U` → `EXECUTORCH_BUILD_ARM_BAREMETAL` (Ethos-U delegate)
3. `add_subdirectory()` ExecuTorch's own CMake, then re-apply Zephyr's compile options to each
   ExecuTorch target (`fix_executorch_target_flags`), so they build with the board's arch flags.
4. Wrap the result in a Zephyr library, `libexecutorch`.

The application (`zephyr/samples/hello-executorch/CMakeLists.txt`) then:

- reads the operator list out of the `.pte` (`codegen/tools/gen_oplist.py`) and builds **only
  those CPU kernels** (selective build, `CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n`);
- converts the `.pte` into `model_pte.h` (`pte_to_header.py`);
- links `libexecutorch`, the selected ops library, `cortex_m_kernels`, `quantized_kernels`,
  `executorch_delegate_ethos_u` and `ethosu_core_driver`.

| Kconfig | Default | Effect |
| :--- | :--- | :--- |
| `CONFIG_EXECUTORCH` | n | Build the runtime |
| `CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS` | y | All portable ops; set n for selective build |
| `CONFIG_EXECUTORCH_OPTIMIZE_FOR_SIZE` | y | `-Os` |
| `CONFIG_EXECUTORCH_ENABLE_LOGGING` | y | Runtime logs |
| `CONFIG_EXECUTORCH_ENABLE_EVENT_TRACER` | n | Profiling |
| `CONFIG_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION` | n | Verify the `.pte` flatbuffer |

**Why.** ExecuTorch has its own CMake ecosystem (codegen, flatcc, kernel registration). The
module reuses it instead of rewriting it as Zephyr CMake. That costs a few hacks: global include
of the parent directory, flag re-application, and `CMAKE_SKIP_INSTALL_RULES` in the sample.
The selective build keeps flash small: only kernels the model needs are linked.

**Where.** `executorch/zephyr/CMakeLists.txt`, `executorch/zephyr/Kconfig`,
`executorch/tools/cmake/preset/zephyr.cmake`, the sample's `CMakeLists.txt`.

**When.** At CMake configure time (oplist extraction, backend selection) and at build time
(`.pte` → header). The `.pte` is passed with `-DET_PTE_FILE_PATH=<file>`; changing the model can
change which kernels are built.

---

## 3. Model deployment

**What.**

1. **Host:** a Python `torch.nn.Module` is exported (`torch.export`), quantized to int8 and
   lowered. The backend's partitioner hands the subgraphs it supports to the NPU compiler; the
   result is one `.pte` file (an ExecuTorch flatbuffer program with delegate payloads inside).
   - Arm: `python -m backends.arm.scripts.aot_arm_compiler --model_name=add.py --quantize
     --delegate --target=ethos-u55-128 --output=add.pte`
   - NXP: `python -m examples.nxp.aot_neutron_compile --quantize --delegate -m cifar10`
2. **Build:** `pte_to_header.py` writes the `.pte` as a byte array in the linker section
   `network_model_sec` (`ET_PTE_SECTION`).
3. **Device:** the runner copies the model to SRAM unless `CONFIG_ET_ARM_MODEL_PTE_DMA_ACCESSIBLE=y`
   (model already in memory the NPU can DMA from), then wraps it in a `BufferDataLoader`.

Memory is set by Kconfig in the samples:

| Pool | Kconfig | Use |
| :--- | :--- | :--- |
| Method allocator | `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` | Program metadata, planned tensor memory |
| Temp allocator | `CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE` | Delegate scratch per run (the Neutron backend takes its scratch from here) |
| Zephyr heap | `CONFIG_HEAP_MEM_POOL_SIZE` | `et_pal_allocate()` → `k_malloc()` |

**Why.** The `.pte` is a self-contained, ahead-of-time-planned program: tensor memory is planned
on the host, so the device needs no graph analysis. Embedding it avoids needing a filesystem.

**Where.** Host: `backends/arm/scripts/aot_arm_compiler.py`, `examples/nxp/aot_neutron_compile.py`.
Build: `examples/arm/executor_runner/pte_to_header.py`. Device:
`zephyr/samples/hello-executorch/src/arm_executor_runner.cpp`,
`examples/arm/executor_runner/arm_memory_allocator.cpp`.

**When.** Export runs once per model change, on the host, with PyTorch installed. The header is
regenerated at build time. Loading happens once at boot; inference runs on each `execute()`.

For mpipe: the model could also be read from the SD card at run time with a `BufferDataLoader`
over a RAM buffer; the samples only show the embedded case.

---

## 4. Runtime and inference backend

**What.** Neither PyTorch nor TensorFlow runs on the device. ExecuTorch has its own small C++
runtime that executes the `.pte` program:

```
runtime_init()                                // PAL init
BufferDataLoader loader(model, size)
Program::load(&loader)                        // parse the flatbuffer
program->load_method("forward", &memory_manager)
method.set_input(tensor, i)
method->execute()                             // runs each instruction:
    CPU op      → registered kernel: portable (plain C++), cortex_m (CMSIS-NN), quantized
    delegate op → backend->execute()  (EthosUBackend, NeutronBackend, ...)
method->get_outputs(...)
```

The platform layer for Zephyr (`runtime/platform/default/zephyr.cpp`) is small:
`et_pal_current_ticks()` → `k_uptime_ticks()`, logs via `fprintf(stderr, ...)`, abort via
`_Exit()`, allocation via `k_malloc()`/`k_free()`. It reports a tick-to-ns ratio of 1, so
ExecuTorch's own timings are in ticks, not nanoseconds.

| | TFLM (what the `jpeg_npu` sample uses) | ExecuTorch |
| :--- | :--- | :--- |
| Model format | `.tflite` | `.pte` |
| Authoring | TensorFlow / Keras (or ONNX converted) | PyTorch |
| Device runtime | `MicroInterpreter` | `Program` / `Method` |
| CPU kernels | reference + CMSIS-NN | portable + cortex_m (CMSIS-NN) + quantized |
| NPU hook | custom op (`NeutronGraph`, `ethos-u`) | delegate backend |
| Memory | one tensor arena | method + temp allocators, planned on host |

**Why.** ExecuTorch's goal is to deploy PyTorch models directly, without converting them to
another framework's format. The host does all graph work; the device only replays the plan.

**Where.** `runtime/executor/`, `runtime/platform/default/zephyr.cpp`, kernel libraries under
`kernels/` and `backends/cortex_m/`.

**When.** At boot (`Program::load`, `load_method`, where delegates run their `init()`) and per
inference (`execute()`).

Note on the NXP flow: inside the AOT step, the NXP backend converts each delegated subgraph to a
**TFLite** model in memory and passes it to NXP's `neutron_converter`
(`backends/nxp/backend/neutron_converter_manager.py`). TFLite is only an internal intermediate;
the device never sees it.

---

## 5. NPU interaction and Neutron compatibility

**What.**

*Arm Ethos-U (integrated).* `CONFIG_ETHOS_U=y` builds the Ethos-U delegate. At run time
`EthosUBackend` calls Arm's driver directly (`ethosu_reserve_driver()`, `ethosu_invoke_v3()`,
`ethosu_release_driver()`), the same driver API and Zephyr device (`drivers/misc/ethos_u`) TFLM
uses. Boards: Corstone-300/320 FVP, Alif Ensemble E8. STM32N6 and Arduino Uno Q run CPU-only.

*NXP Neutron (not integrated with Zephyr).* ExecuTorch has a Neutron backend:

- **AOT:** `NeutronQuantizer` + `NeutronPartitioner` + `neutron_converter` from the pip package
  `eiq-neutron-sdk` (`backends/nxp/requirements-eiq.txt` pins **3.1.3** in v1.4.0).
- **Runtime:** `backends/nxp/runtime/NeutronBackend.cpp`, a `PyTorchBackendInterface`:
  - `init()`: checks the microcode magic, finds microcode/weights/kernels in the delegate
    payload, calls `neutronModelPrepare()`.
  - `execute()`: allocates scratch from the temp allocator, transposes NCHW inputs to NHWC on the
    CPU when needed, `neutronResume()` → `neutronRunBlocking()` → `neutronSuspend()`, transposes
    outputs back.
  - `destroy()`: `neutronModelUnprepare()`.

  These are the same driver calls the TFLM `neutron.cpp` kernel makes.

NXP's supported deployment of it is **MCUXpresso SDK 26.06.00** on i.MX RT700, not Zephyr.

**Why it does not work on Zephyr yet.**

| Gap | Detail |
| :--- | :--- |
| Not wired in the module | `zephyr/CMakeLists.txt` sets only `CORTEX_M`, `XNNPACK`, `ARM_BAREMETAL`; it never sets `EXECUTORCH_BUILD_NXP_NEUTRON`, so `executorch_delegate_neutron` is never built |
| No driver linkage | The backend's CMake links no driver; the NXP runner links the **cmodel** (simulator) driver. On target it must link `neutron_driver`/`neutron_firmware` (hal_nxp or SDK) |
| Version lock | AOT microcode from `eiq-neutron-sdk` **3.1.3**; `hal_nxp` blobs are **3.0.0**; your SDK is **3.2.3**. All three must be the same version, or `neutronModelPrepare()` fails |
| Header mismatch | ExecuTorch bundles its own `NeutronDriver.h` (2022-2024). Its `NeutronModelConfig` ends at `timeoutSeconds`; the 3.x headers (hal_nxp and SDK) add `const char *subgraphName`. A 3.x driver reading that field gets whatever follows in memory (inferred from the headers; not tested) |
| Memory and cache | Scratch comes from the temp allocator, which must sit in NPU-reachable, non-cacheable SRAM; the `.pte` itself must be in NPU-reachable SRAM (or use the `EXTERNAL_MEM` copy path) |
| Heap | The bundled header defaults to `NO_HEAP_USAGE 0`: the driver allocates the model handle itself, so Zephyr needs a heap the blob's `malloc` can use |
| Power | No NPU power-up code is in this path; on RT700 `board.c` does it only when `CONFIG_NXP_NEUTRON=y` |

**Where.** `backends/nxp/` (AOT and runtime), `examples/nxp/` (AOT script, cmodel runner),
`zephyr/CMakeLists.txt` (missing hook), `hal_nxp/zephyr/module.yml` (blob version).

**When.** For a Neutron-on-Zephyr port, the work is:

1. Add an `EXECUTORCH_BUILD_NXP_NEUTRON` hook to the module CMake (for example on
   `CONFIG_NXP_NEUTRON`), and link `executorch_delegate_neutron` with `neutron_driver` and
   `neutron_firmware`.
2. Build against the **3.x** `NeutronDriver.h` (hal_nxp or SDK), not the bundled copy.
3. Use one Neutron version everywhere: AOT `eiq-neutron-sdk`, runtime blobs and header (3.2.3 with
   your SDK, overriding the hal_nxp blobs as `jpeg_npu` does).
4. Place the `.pte`, method pool and temp pool in NPU-reachable, non-cacheable SRAM (SRAM1).
5. Validate with `examples/nxp` CifarNet, then MobileNetV2.

Until then, on RT700 under Zephyr, ExecuTorch can only run models on the Cortex-M33
(portable / CMSIS-NN kernels), while TFLM + Neutron (the `jpeg_npu` sample) already uses the NPU.

---

## Sources

ExecuTorch v1.4.0 (commit `3dd7ccd1d863`):
`zephyr/{module.yml,executorch.yaml,Kconfig,CMakeLists.txt,README.md}`,
`zephyr/samples/{hello-executorch,mv2-ethosu}/`, `tools/cmake/preset/zephyr.cmake`,
`runtime/platform/default/zephyr.cpp`, `backends/nxp/{CMakeLists.txt,requirements-eiq.txt}`,
`backends/nxp/runtime/{NeutronBackend.cpp,NeutronDriver.h}`,
`backends/nxp/backend/neutron_converter_manager.py`, `backends/arm/runtime/EthosUBackend_Cortex_M.cpp`,
`examples/nxp/{README.md,executor_runner/CMakeLists.txt}`, top-level `CMakeLists.txt`.

Web: [Zephyr: ExecuTorch external module](https://docs.zephyrproject.org/latest/develop/manifest/external/executorch.html),
[ExecuTorch NXP backend overview](https://docs.pytorch.org/executorch/1.2/backends/nxp/nxp-overview.html).

Workspace: `modules/hal/nxp/zephyr/module.yml` (Neutron blobs 3.0.0),
`/home/linux/Downloads/eiq-neutron-sdk-linux-3.2.3/target/imxrt700/driver/include/NeutronDriver.h`.
