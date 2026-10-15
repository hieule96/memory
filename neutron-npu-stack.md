# NXP Neutron NPU in Zephyr: how it works, how to use it, and how it compares

Status as of 2026-10-13. Based on the `mpipe_dev` Zephyr workspace (`hal_nxp`, `hal_ethos_u`,
`samples/boards/nxp/tflm_neutron`) and on public documentation listed under [Sources](#sources).
Nothing described here has been built or run on hardware. The Neutron blobs are not fetched in
this workspace yet (`west blobs fetch hal_nxp`).

## Contents

1. [Summary](#1-summary)
2. [Can ExecuTorch run on Neutron in Zephyr?](#2-can-executorch-run-on-neutron-in-zephyr)
3. [How the Neutron NPU works](#3-how-the-neutron-npu-works)
4. [How Neutron is integrated in Zephyr](#4-how-neutron-is-integrated-in-zephyr)
5. [One inference traced, application to NPU](#5-one-inference-traced-application-to-npu)
6. [Practical constraints](#6-practical-constraints)
7. [The NPU interface: what is public](#7-the-npu-interface-what-is-public)
8. [Using Neutron without TFLM or ExecuTorch](#8-using-neutron-without-tflm-or-executorch)
9. [The common NPU software stack](#9-the-common-npu-software-stack)
10. [Other NPU families compared](#10-other-npu-families-compared)
11. [Does Neutron go through a Zephyr driver API?](#11-does-neutron-go-through-a-zephyr-driver-api)
12. [Why only Ethos-U has OS-level glue](#12-why-only-ethos-u-has-os-level-glue)
13. [What this means for mpipe](#13-what-this-means-for-mpipe)
14. [Verified vs not verified](#14-verified-vs-not-verified)
15. [Sources](#sources)

---

## 1. Summary

- Neutron is reached through NXP's closed driver library (`libNeutronDriver.a`) and its C API
  `NeutronDriver.h`. Zephyr only enables it (Kconfig), links the blobs and powers up the NPU in
  board code. There is no Zephyr device, devicetree node or interrupt for it.
- TensorFlow Lite Micro (TFLM) uses Neutron through one custom operator, `NeutronGraph`, which
  calls the driver. ExecuTorch has its own NXP backend that calls the same driver. Using it in
  Zephyr is possible but not available off the shelf.
- The application can also call the driver directly, without any framework, as long as the
  model is compiled by NXP's `neutron-converter`.
- Every MCU NPU uses the same six-layer stack (model, compiler, framework, glue, driver,
  hardware). Only the model formats, the frameworks (TFLM, ExecuTorch) and the CPU kernels
  (CMSIS-NN) are shared across NPU families. Compiler, driver and firmware are vendor-specific.
- Arm Ethos-U is the only NPU with a real Zephyr device driver, because its driver is open
  source and built to be ported to any OS through weak-symbol hooks.

---

## 2. Can ExecuTorch run on Neutron in Zephyr?

Yes, with integration work. Nothing in the tree does it today.

What exists:

- `CONFIG_NXP_NEUTRON` (in `zephyr/modules/hal_nxp/mcux/Kconfig.mcux`) is independent of TFLM.
  It links `libNeutronDriver.a` and `libNeutronFirmware.a` into the application. Blobs exist only
  for RT700 (CM33 CPU0) and MCXN947.
- ExecuTorch has an NXP backend (`backends/nxp/`). On the host, a partitioner and quantizer hand
  supported subgraphs toa k_mutex around runs, so two threads can't use the NPU at once,
 the Neutron converter and embed the compiled result in the `.pte` file.
  On the device, a `NeutronBackend` delegate runs that subgraph through the same Neutron driver
  API. Unsupported operators fall back to CPU kernels.
- ExecuTorch 1.2 moved the NXP backend to the self-contained eIQ Neutron SDK, and its docs list
  i.MX RT700 as the supported platform, with deployment through MCUXpresso projects.
- ExecuTorch can be added to Zephyr as an external west module. Zephyr's documentation covers
  the Arm Ethos-U backend and a CPU-only path. It does not cover the NXP backend.

Steps to make it work:

1. Add ExecuTorch as a west module and build its runtime as a Zephyr library: core runtime,
   portable or Cortex-M kernels, and the `NeutronBackend` delegate. Disable host-only features.
2. Set `CONFIG_NXP_NEUTRON=y` and link the delegate against the `neutron_driver` and
   `neutron_firmware` targets from `hal_nxp`. Do not pull a second driver copy from MCUXpresso.
3. Match versions: the converter used to export the `.pte` must match the driver and firmware
   blobs (v3.0.0 in `hal_nxp/zephyr/module.yml`). This is the most likely failure point.
4. Provide static memory: method allocator, planned-memory arena, the `.pte` in flash, aligned
   scratch. Add a minimal platform layer (log, time) mapped to `printk` and `k_uptime`.
5. Target `mimxrt700_evk/mimxrt798s/cm33_cpu0` or an MCXN947 board.

---

## 3. How the Neutron NPU works

Neutron does not interpret a model graph. It runs a program compiled offline for one model.

| Piece | Comes from | What it is |
| :--- | :--- | :--- |
| Microcode | `neutron-converter`, embedded in the model | Program for the NPU: layer sequence, data movement, synchronization |
| Weights | `neutron-converter` | Weights rearranged into the layout the compute units need |
| Kernels | `neutron-converter` | Compute routines for the layer types the model uses |
| Firmware | `libNeutronFirmware.a` | NXP code running on the NPU's controller; executes the microcode |
| Driver | `libNeutronDriver.a` | Runs on the Cortex-M33: starts the firmware, passes buffers, starts and waits for runs |

The header states that microcode, weights and kernels are 16-byte aligned and "passed by
reference to the Neutron firmware". The same holds for inputs, outputs and scratch: the NPU reads
and writes them in memory by address, with no copy by the CPU.

The converter takes a quantized `.tflite` and outputs a `.tflite` in which the supported layers
are replaced by one `NeutronGraph` custom operator. Layers it cannot map stay as normal TFLM
operators running on the CPU.

---

## 4. How Neutron is integrated in Zephyr

| Layer | Location | Role |
| :--- | :--- | :--- |
| Kconfig | `zephyr/modules/hal_nxp/mcux/Kconfig.mcux` | `NXP_NEUTRON` (RT7xx, MCXN), `TFLM_NXP_NEUTRON` selects it |
| CMake | `hal_nxp/mcux/middleware/neutron/CMakeLists.txt` | Imports the two blobs per SoC, builds the TFLM kernel, links into `app` |
| Blobs | `hal_nxp/zephyr/module.yml` | Driver and firmware v3.0.0, from NXP `mcuxsdk-middleware-eiq` |
| Driver header | `hal_nxp/mcux/middleware/neutron/include/NeutronDriver.h` | The public API |
| Framework glue | `hal_nxp/mcux/middleware/neutron/tflite-micro/.../neutron/neutron.cpp` | TFLM custom op `NeutronGraph` |
| Power | `zephyr/boards/nxp/mimxrt700_evk/board.c` | `POWER_DisablePD(kPDRUNCFG_APD_NPU / PPD_NPU)` when `CONFIG_NXP_NEUTRON` |
| Sample | `zephyr/samples/boards/nxp/tflm_neutron/` | Image classification with TFLM and Neutron |

Not present: devicetree binding or node, `struct device`, `IRQ_CONNECT`, power management, cache
or locking hooks. No NPU power-up code was found in the MCXN board files.

Note: the CMake adds `NeutronDriver.h`'s include path only to the TFLM target. An application
that uses the driver without TFLM must add it itself.

---

## 5. One inference traced, application to NPU

```
 HOST (once)
 model.tflite (int8) --neutron-converter--> model_npu.tflite
     supported layers -> one "NeutronGraph" op + 3 constant inputs:
                         microcode, weights, kernels
     other layers     -> stay as TFLM ops (Dequantize, Reshape, Softmax...)
 xxd -i -> g_model[] compiled into flash

 TARGET
 Application (main_functions.cpp)
   setup(): GetModel(g_model)
            resolver.AddCustom("NeutronGraph", Register_NEUTRON_GRAPH())
            MicroInterpreter(model, resolver, tensor_arena)
            AllocateTensors()  -----------------> TFLM: Init + Prepare of each op
   loop():  fill input tensor
            Invoke()  ------------------------> TFLM: Eval of each op in order
                                                   |
 Framework glue (neutron.cpp)                      v
   Init:    neutronInit()                    (first instance only, ref-counted)
   Prepare: model_config = {microcode, weights, kernels}
            model context placed in the TFLM arena
            neutronModelPrepare(&cfg, &handle)   <- version check here
   Eval:    data_config.inputs[]  = input tensor addresses
            data_config.outputs[] = output tensor addresses
            data_config.scratch   = scratch tensor
            neutronRunBlocking(handle, &data_config)
   Free:    neutronModelUnprepare(); neutronDeinit() (last instance)
                                                   |
 Driver blob (Cortex-M33)                          v
   hands addresses to the firmware, starts the NPU, waits
                                                   |
 NPU + firmware blob                               v
   runs the microcode: reads inputs and weights, writes outputs and scratch
                                                   |
 Back in TFLM: CPU ops (here Dequantize) run on the Cortex-M33; application reads the output
```

Sequence of calls:

```
 App            TFLM             Neutron op         Driver          NPU
  |-- AllocateTensors() -->|                          |               |
  |                |-- Init, Prepare -->|             |               |
  |                |                |-- neutronInit() -->|           |
  |                |                |-- neutronModelPrepare() -->|   |
  |                |                |                 |-- params -->|
  |  (fill input)  |                |                 |              |
  |-- Invoke() --->|                |                 |              |
  |                |-- Eval ------->|                 |              |
  |                |                |-- neutronRunBlocking() -->|    |
  |                |                |   [calling thread  |-- start ->|
  |                |                |    blocked]       |   runs microcode
  |                |                |                 |<-- done ----|
  |                |                |<-- ENONE -------|              |
  |                |<-- kTfLiteOk --|                 |              |
  |                | Dequantize on CPU                |              |
  |<-- output -----|                |                 |              |
```

---

## 6. Practical constraints

- **Cache coherency.** The NPU reads and writes memory directly, bypassing the CM33 data cache.
  The sample places `tensor_arena` in `__nocache` memory on RT700 for this reason. Any buffer
  passed in `NeutronDataConfig` (for example an mpipe `net_buf`) must be non-cacheable, or be
  flushed before the run and invalidated after.
- **Weights in flash.** Weights are passed by address and read from flash. `neutronSetConfig()`
  lets the application supply DMA `copy`/`wait` callbacks to prefetch weights into SRAM
  (`EXTERNAL_MEM` branch in `neutron.cpp`).
- **Version lock.** Converter and blob versions must match. A mismatch fails at run time in
  `neutronModelPrepare()` with `ERROR_CATEGORY_DRIVER_UCODE`, not at build time. The sample
  README cites eIQ Toolkit `MCU_SDK_25.03.00`; the blobs are v3.0.0.
- **Blocking call.** `neutronRunBlocking()` holds the calling thread for the whole run. To
  overlap work, run inference in its own thread.
- **No locking.** Nothing serializes NPU access between threads. Two threads running models need
  a `k_mutex` around the run.

---

## 7. The NPU interface: what is public

### Driver API (`NeutronDriver.h`)

This is the only supported interface. It works on whole compiled models.

| Function | Purpose | On RT700 / MCXN |
| :--- | :--- | :--- |
| `neutronInit()`, `neutronDeinit()` | Start or stop driver and firmware | Yes |
| `neutronGetModelContextSize()` | Size of the per-model context, so the caller can provide it | Yes |
| `neutronModelPrepare(cfg, &hdl)` | Register `{microcode, weights, kernels, timeoutSeconds, subgraphName}`; version check | Yes |
| `neutronModelUnprepare(hdl)` | Release the model context | Yes |
| `neutronDataSetup(hdl, dcfg)` | Point inputs and outputs at the NPU's own memory area | Not used by TFLM |
| `neutronRunBlocking(hdl, dcfg)` | Run one inference on `{inputs[], outputs[], scratch, scratchWeights}` | Yes |
| `neutronRunNonBlocking()`, `neutronWait()`, `neutronIsReady()` | Asynchronous run | Header: Neutron-S only |
| `neutronCustomPrepare()`, `neutronCustomExec()` | Run custom firmware | Header: Neutron-S only |
| `neutronSuspend()`, `neutronResume()` | Gate the NPU clock | Yes |
| `neutronSetConfig({copy, wait})` | DMA hooks for weight prefetch | Yes |
| `neutronMemAlloc()`, `allocateBuffer()`, `clean_cache()`, `matmul()` | Linux / DDR helpers | Linux / i.MX only |

Model context: passing a pointer to caller-owned memory as the handle avoids the driver's heap
("hdl shall be a pointer to the start of the allocated memory block"). Passing NULL makes the
driver allocate from the heap.

Errors: `NeutronError` packs a component (library, firmware, driver), a category (`UCODE`,
`NO_MEMORY`, `TIMEOUT`, `BAD_HANDLE`, ...) and a code. `getNeutronErrorComponent()` and
`getNeutronErrorCategory()` decode it.

Asynchronous runs are documented as Neutron-S only. On RT700 and MCXN, assume only
`neutronRunBlocking()` until tested (an unsupported call should return
`ERROR_CATEGORY_DRIVER_UNSUPPORTED`).

### Hardware

- **RT700, MCXN:** no register map, no base address in the SoC headers. The RT700 header has
  `NPU_IRQn = 141`; Zephyr does not connect it. MCXN947 has no NPU entry at all.
- **i.MX95:** the register map is public (`mcux-sdk-ng/devices/i.MX/i.MX95/periph/PERI_NEUTRON*.h`)
  and shows the design:

```
         SoC CPU (driver)                         Neutron subsystem
 +--------------------------+  MBOX[], HEAD/TAIL ring, +--------------------------------+
 | neutronRunBlocking()     |--INPUT/OUTPUT/CODEOFF--->| Zen-V (NXP RISC-V controller)  |
 |                          |<-APPSTATUS, INTENA/INTCLR| runs libNeutronFirmware,       |
 +--------------------------+                          | reads microcode, programs:     |
                                                       |                                |
                 system RAM / DDR / flash              | Data mover   DSADDR/DDADDR     |
          +------------------------+  fetch/push       | Weight fetch WSADDR/WDADDR,    |
          | inputs, outputs, spill,|<----------------->|   DECOMPCTRL (decompressor)    |
          | weights                |                   | Compute pipeline: WEIGHTBASE,  |
          +------------------------+                   |   BIASBASE, SCALEBASE, ACTCTRL,|
                                                       |   CANVAS_* tiling, RESBASE     |
                                                       | Perf counters CYCLOW, NACT     |
                                                       +--------------------------------+
```

A RISC-V controller runs the firmware. It programs DMA engines and a fixed-function compute
pipeline layer by layer, as the microcode says. The host talks to it through mailboxes, a ring
buffer and status registers. That the MCU variants follow the same design is an inference; their
registers are not published.

Consequence: on RT700 and MCXN you cannot program the NPU directly. The microcode format,
firmware and register map are not public, and they change between versions.

---

## 8. Using Neutron without TFLM or ExecuTorch

Possible: call `NeutronDriver.h` directly. The converter is still required.

### What the converted model contains

Parsed from `samples/boards/nxp/tflm_neutron/src/models/rt700/model.hpp` (CifarNet):

```
op NeutronGraph
  in  #0 'input'              i8[1,32,32,3]   scale=0.0078125   zero point=0
  in  #3 'NeutronMicrocode'   u8[1368]        constant
  in  #4 'NeutronWeights'     u8[97440]       constant
  in  #5 'NeutronKernels'     u8[1]           constant
  out #2 '..._Reshape_1_int8' i8[1,10]        scale=0.00390625  zero point=-128
  out #6 'NeutronScratch'     u8[40960]
  out #7 'NeutronProfile'     u8[0]
  out #8 'NeutronDebug'       u8[0]
op DEQUANTIZE (CPU)  #2 -> #1 f32[1,10]
```

The whole network is offloaded except the final Dequantize.

### What the application takes over

| TFLM did | Without a framework |
| :--- | :--- |
| Find microcode, weights, kernels in the flatbuffer | Extract them offline into 16-byte-aligned `const` arrays |
| Place tensors in the arena | Static buffers: input 3072 B, output 10 B, scratch 40960 B (non-cacheable on RT700) |
| Model context in the arena | Static buffer, checked against `neutronGetModelContextSize()` |
| Build `inputs[]` / `outputs[]` | Copy the layout used by `neutron.cpp` |
| Run CPU ops | Write them (here `(q + 128) * 0.00390625`) |
| Quantize input | `int8 = round(x / 0.0078125)` for this model |

### Build

`prj.conf`:

```
CONFIG_NXP_NEUTRON=y
```

`CMakeLists.txt`:

```cmake
target_include_directories(app PRIVATE
	${ZEPHYR_HAL_NXP_MODULE_DIR}/mcux/middleware/neutron/include)
```

### Code sketch

```c
#include <zephyr/kernel.h>
#include <zephyr/linker/section_tags.h>

#include "NeutronDriver.h"
#include "cifarnet_neutron.h" /* microcode[], weights[], kernels[], generated offline */

#define MODEL_CTX_SIZE 1024U /* checked against neutronGetModelContextSize() */

static int8_t input[32 * 32 * 3] __nocache __aligned(16);
static int8_t output[10] __nocache __aligned(16);
static uint8_t scratch[40960] __nocache __aligned(16);
static uint8_t model_ctx[MODEL_CTX_SIZE] __aligned(16);

/* Same layout as neutron.cpp: real outputs, extra tensors, scratch last */
static const void *inputs[] = {input};
static void *outputs[] = {output, scratch, NULL, scratch};

int main(void)
{
	NeutronModelConfig mcfg = {
		.microcode = microcode,
		.weights = weights,
		.kernels = kernels,
	};
	NeutronDataConfig dcfg = {
		.inputs = inputs,
		.outputs = outputs,
		.scratch = scratch,
	};
	NeutronModelHandle hdl = model_ctx;
	NeutronError err;

	if (neutronGetModelContextSize() > sizeof(model_ctx)) {
		printk("Raise MODEL_CTX_SIZE to %zu\n", neutronGetModelContextSize());
		return -ENOMEM;
	}

	err = neutronInit();
	if (err != ENONE) {
		printk("neutronInit failed (0x%x)\n", err);
		return -EIO;
	}

	err = neutronModelPrepare(&mcfg, &hdl);
	if (err != ENONE) {
		/* ERROR_CATEGORY_DRIVER_UCODE: converter and blob versions differ */
		printk("neutronModelPrepare failed (0x%x)\n", err);
		return -EIO;
	}

	/* Fill input[] with the quantized image */

	err = neutronRunBlocking(hdl, &dcfg);
	if (err != ENONE) {
		printk("neutronRunBlocking failed (0x%x)\n", err);
		return -EIO;
	}

	for (size_t i = 0; i < ARRAY_SIZE(output); i++) {
		float score = (output[i] + 128) * 0.00390625f;
		/* ... */
	}

	return 0;
}
```

### Risks

- The `outputs[]` layout (real outputs, extra tensors, scratch last) is copied from `Eval` in
  `neutron.cpp`, which fills every slot but the last with tensor pointers and puts scratch in
  the last slot. It is not documented and may change with a converter release.
- `NeutronProfile` and `NeutronDebug` are 0 bytes; passing NULL for them is untested.
- A model the converter splits into several `NeutronGraph` ops needs one handle and one run per
  op, plus your own CPU code between them.
- The blob's own dependencies (libc, C++ runtime) are unknown until link time.

---

## 9. The common NPU software stack

Every MCU NPU looked at uses the same layers. Neutron shown on the right.

```
             +------------------------------------------------------------------+
  Host,      | 1  Model             int8-quantized network   CifarNet .tflite    |
  build time +------------------------------------------------------------------+
             | 2  Offline compiler  maps layers to the NPU   neutron-converter   |
             |                      and emits its binary     -> NeutronGraph op  |
             +------------------------------------------------------------------+
             +------------------------------------------------------------------+
  Target,    | 3  Framework/runtime loads model, plans       TFLM, ExecuTorch    |
  run time   |                      memory, runs ops         |                   |
             +-------------------------------+----------------------------------+
             | 4a NPU glue                   | 4b CPU kernels                   |
             |    op or delegate calling     |    ops the NPU cannot run        |
             |    the driver: NeutronGraph   |    CMSIS-NN                      |
             +-------------------------------+                                  |
             | 5  Driver                     |                                  |
             |    libNeutronDriver.a         |                                  |
             +-------------------------------+----------------------------------+
             | 6  Firmware + NPU hardware    |    Cortex-M CPU                  |
             |    libNeutronFirmware.a       |    runs layers 3, 4a, 4b, 5      |
             +-------------------------------+----------------------------------+
```

---

## 10. Other NPU families compared

| NPU (silicon) | Offline compiler | Framework | Glue to the NPU | Driver / firmware | In upstream Zephyr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NXP Neutron (RT700, MCXN947) | `neutron-converter` (eIQ) | TFLM; ExecuTorch NXP backend | `NeutronGraph` op; `NeutronBackend` | Closed blobs, model-level C API | Kconfig, blobs, board power-up, TFLM sample |
| Arm Ethos-U55/U65/U85 (Corstone, Alif, Renesas RA, Infineon PSE84, Nuvoton M55M1) | Vela (open source) | TFLM; ExecuTorch Arm backend | TFLM `ethos-u` op; Ethos-U delegate | Open `ethos-u-core-driver` | Full device driver (`drivers/misc/ethos_u`), `arm,ethos-u` binding + vendor frontends, `hal_ethos_u`, ExecuTorch external module |
| ST Neural-ART (STM32N6) | ST Edge AI Core: generated C code + command blobs | None (generated code is the runtime) | Generated network code | `ll_ATON` runtime + ST CPU operator library | `st,stm32-npu` node; `soc/st/stm32/stm32n6x/npu` does clock, reset, NPU cache |
| Nordic Axon (nRF54LM20B) | Nordic Edge AI tools; Nordic says it accelerates TFLite models | Not documented in sources found | Not documented | Not public | `nordic,axon` binding and node only |
| TI TinyEngine (MSPM0G5187, C2000 F28P55x) | TI Neural Network Compiler (Apache TVM based), outputs `.h` + `.a` | None (compiled library is the runtime) | Compiled library | Inside the generated library | None |

### Shared vs vendor-specific

```
                 NXP Neutron   Arm Ethos-U   ST Neural-ART  Nordic Axon   TI TinyEngine
 Model        [====== TensorFlow Lite int8 (.tflite) ======================]  TI Tensorlab
              [== PyTorch (.pte) =========]                                  model
 Compiler      neutron-conv  Vela (open)   ST Edge AI     Nordic tools   TI NNC (TVM)
 Framework    [== TFLM ===================]  +-----------+  +----------+  +-----------+
              [== ExecuTorch =============]  | generated |  | not      |  | generated |
 NPU glue      NeutronGraph  ethos-u op    | C code +  |  | public   |  | inference |
 CPU kernels  [== CMSIS-NN ===============]  | ll_ATON + |  |          |  | library   |
 Driver/fw     closed blobs  open driver   | ST CPU lib|  |          |  | (.h + .a) |
                                             +-----------+  +----------+  +-----------+
 Zephyr        Kconfig,      device driver, DT node,       DT node only   none
               blobs, power  DT, IRQ        clock/reset/cache

 [=== ===] shared by more than one NPU family; plain entries are vendor-specific
```

The shared layers are above the NPU: model formats, TFLM and ExecuTorch, and CMSIS-NN. Below
them each vendor ships its own compiler and runtime, and each compiler's output runs only on that
vendor's NPU. ST and TI skip the framework layer: their compilers generate the runtime as C code.

---

## 11. Does Neutron go through a Zephyr driver API?

No. The application or the TFLM op calls NXP's driver functions directly. Zephyr has no generic
NPU driver class (`include/zephyr/drivers/` has none).

```
 Application / TFLM NeutronGraph op
        |  neutronRunBlocking(...)        plain C call into the blob
        v
 libNeutronDriver.a (NXP, closed)         no struct device, no Zephyr API
        |
        v
 Neutron NPU + libNeutronFirmware.a
```

Ethos-U, for comparison:

```
 Application / TFLM ethos-u op
        |  ethosu_reserve_driver(), ethosu_invoke_v3(...)   Arm's API, not a Zephyr API
        v
 ethos-u-core-driver (Arm, open source, hal_ethos_u)
        |  calls OS hooks the platform provides
        v
 Zephyr glue: drivers/misc/ethos_u/
   ethos_u_arm.c     DEVICE_DT_INST_DEFINE from "arm,ethos-u", IRQ_CONNECT -> ethosu_irq_handler(),
                     ethosu_init() at boot
   ethos_u_common.c  ethosu_mutex_*     -> k_mutex
                     ethosu_semaphore_* -> k_sem (the inference thread sleeps until the IRQ)
                     ethosu_flush/invalidate_dcache -> sys_cache_data_flush/invd_range
```

| | Neutron | Ethos-U |
| :--- | :--- | :--- |
| Inference API | NXP `neutron*()` (blob) | Arm `ethosu_*()` (open) |
| Zephyr device, devicetree | No | Yes (`arm,ethos-u`) |
| Interrupt via Zephyr | No (polled or hidden in the blob) | Yes, thread sleeps on `k_sem` |
| Cache maintenance via Zephyr | No (sample uses `__nocache`) | Yes, `sys_cache_*` hooks |
| Locking between threads | None visible | `k_mutex` per driver |

In both cases the inference call is the vendor's API. The Ethos-U Zephyr device only sets up the
hardware and implements the driver's OS hooks.

---

## 12. Why only Ethos-U has OS-level glue

Other NPUs have some glue, at different levels:

| NPU | Glue in Zephyr | What it connects |
| :--- | :--- | :--- |
| Ethos-U | `drivers/misc/ethos_u/` (2022, by Arm) | Device, devicetree, IRQ, `k_mutex`/`k_sem`, `sys_cache_*` |
| NXP Neutron | Kconfig, blob linking, `board.c` power-up (sample 2026-05) | Power only; `neutron.cpp` sits above the driver |
| ST Neural-ART | `soc/st/stm32/stm32n6x/npu/` (2025-07) | Clock, reset, NPU cache at boot |
| Nordic Axon | `nordic,axon` devicetree node | Nothing yet upstream |

Reasons:

1. **Business model.** Arm licenses Ethos-U to many silicon vendors running many operating
   systems, so it ships one portable, open-source driver with a public register map. The OS
   dependencies are weak symbols in `ethosu_driver.c` (`ethosu_mutex_*`, `ethosu_semaphore_*`,
   `ethosu_flush_dcache`, `ethosu_invalidate_dcache`), and `ethosu_irq_handler()` is exported
   for the platform's ISR. The Zephyr glue only overrides them, and one glue serves every vendor
   (Nuvoton, Renesas and Infineon add thin devicetree frontends). Neutron, Neural-ART and Axon are
   each used only in their vendor's chips, which ship software through their own SDKs
   (MCUXpresso, STM32Cube, nRF Connect SDK, MSPM0 SDK).
2. **A closed driver exposes nothing to connect.** `NeutronDriver.h` has no lock or semaphore
   hooks, no interrupt entry point and no cache callbacks. The only hook is `neutronSetConfig()`
   for weight-prefetch DMA. Zephyr can wrap the blob with power, clock and locking, but cannot
   make the thread sleep on the NPU interrupt unless the blob offers an entry point. ST's
   `ll_ATON` is source, but it is generated with each model and written for STM32Cube.
3. **Upstream effort.** Arm engineers wrote the Zephyr Ethos-U driver in November 2022 and
   maintain `hal_ethos_u` and `drivers/misc/ethos_u/` (`MAINTAINERS.yml`). ST, NXP and Nordic
   added their NPU support later and with less scope.

A Zephyr-style Neutron driver could add, without NXP changes: a devicetree binding, power and
clock handling with `neutronSuspend()`/`neutronResume()` tied to Zephyr power management, a
`k_mutex` around runs, and cache maintenance. Sleeping on the interrupt would need NXP to expose an
interrupt entry point or release the driver source. Checking the blob's symbols
(`arm-none-eabi-nm libNeutronDriver.a`) would show whether one exists.

---

## 13. What this means for mpipe

- The stable integration point is the framework layer: TFLM (and later ExecuTorch) already hides
  Neutron and Ethos-U behind one API.
- Calling the Neutron driver directly fits mpipe's rules better: no C++, no interpreter, all
  memory static and sized offline. An inference transform element would own the handle and
  buffers, call `neutronModelPrepare()` on READY -> PAUSED, and in its `chain_fn` convert the
  frame into the input buffer, call `neutronRunBlocking()` and pass the result downstream.
- Buffers handed to the NPU must be non-cacheable or cache-maintained, and the element must lock
  the NPU if more than one pipeline can use it.
- ST and TI bypass the framework layer, so supporting them would need a separate element.

---

## 14. Verified vs not verified

Verified from source in this workspace:

- Neutron Kconfig, CMake, blob versions, driver header, TFLM kernel, sample, RT700 power-up.
- The converted CifarNet model's operators, tensors and quantization (parsed from `model.hpp`).
- i.MX95 Neutron register headers; RT700 `NPU_IRQn = 141`.
- Ethos-U driver API, weak hooks, Zephyr device driver, maintainers and history.
- ST NPU SoC init, Nordic Axon devicetree node, absence of a Zephyr NPU driver class.

From public documentation (search summaries and one fetched page):

- ExecuTorch NXP backend status and platforms; ExecuTorch as a Zephyr module.
- ST Edge AI Core and `ll_ATON`; Nordic Axon tooling; TI Neural Network Compiler.

Not verified:

- Anything inside the Neutron blobs (interrupt or polling, dependencies, symbols).
- Whether `neutronRunNonBlocking()` works on RT700 or MCXN.
- Whether the driver accepts NULL for zero-size output tensors.
- That RT700/MCXN Neutron follows the i.MX95 design.
- Nordic Axon's runtime and where its driver lives.
- No code was built or run; no hardware was used.

---

## 15. Possible to quantize using other than eiQ and remove eiQ complete

No, because the eiQ has the neutron compiler that compile a model into binary of the execution code that the neutron compiler can convert. The library is closed source and not open source.

Moreover the neutron can only accept the input tflite for now ? 
- For now it is not possible to use any format rather than tflite.

---

### 16. 

## Sources

Workspace:

- `modules/hal/nxp/mcux/middleware/neutron/` (driver header, TFLM kernel, CMake)
- `modules/hal/nxp/zephyr/module.yml` (blob versions)
- `modules/hal/nxp/mcux/mcux-sdk-ng/devices/i.MX/i.MX95/periph/PERI_NEUTRON*.h`
- `modules/hal/nxp/mcux/mcux-sdk-ng/devices/RT/RT700/MIMXRT798S/MIMXRT798S_cm33_core0_COMMON.h`
- `zephyr/modules/hal_nxp/mcux/Kconfig.mcux`
- `zephyr/samples/boards/nxp/tflm_neutron/`
- `zephyr/boards/nxp/mimxrt700_evk/board.c`
- `modules/hal/ethos_u/` and `zephyr/drivers/misc/ethos_u/`
- `zephyr/dts/bindings/arm/{arm,ethos-u,nuvoton,numaker-npu,renesas,ra-npu,infineon,edge-npu}.yaml`
- `zephyr/dts/bindings/misc/{st,stm32-npu,nordic,axon}.yaml`
- `zephyr/soc/st/stm32/stm32n6x/npu/`
- `zephyr/MAINTAINERS.yml`

Web:

- [ExecuTorch NXP backend overview (1.2)](https://docs.pytorch.org/executorch/1.2/backends/nxp/nxp-overview.html)
- [ExecuTorch 1.2.0 release notes](https://newreleases.io/project/pypi/executorch/release/1.2.0)
- [Zephyr: ExecuTorch external module](https://docs.zephyrproject.org/latest/develop/manifest/external/executorch.html)
- [ExecuTorch Zephyr tutorial for Alif](https://docs.pytorch.org/executorch/stable/zephyr_alif_tutorial.html)
- [ST Neural-ART compiler primer](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_neural_art_compiler.html)
- [Embassy STM32N6 NPU notes](https://docs.embassy.dev/embassy-stm32/git/stm32n647a0/npu/index.html)
- [Nordic Axon NPU](https://www.nordicsemi.com/Products/Technologies/Edge-AI/Axon-NPU)
- [TI MSPM0G5187](https://www.ti.com/product/MSPM0G5187)
- [TI Neural Network Compiler for MCUs: compilation](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/compiling.html)

A version with rendered diagrams is in the doc
[Edge NPU Software Stacks](https://claude.ai/code/artifact/7042a283-437b-489c-9850-f293d59d4b0c).
