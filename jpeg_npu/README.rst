.. zephyr:code-sample:: mpipe-jpeg-npu
   :name: MJPEG decode and NPU inference pipeline
   :relevant-api: mpipe_img mpipe_fs mpipe_base

   Decode a JPEG sequence from an SD card and classify every frame with a
   TensorFlow Lite Micro model, on the NXP Neutron NPU where available.

Overview
********

This sample reads a Motion-JPEG file (a sequence of JPEG frames) from an SD card, decodes it with
the Multimedia Pipeline subsystem, and runs an image classification model on each decoded frame.

On the i.MX RT700 the JPEG frames are decoded by the hardware JPEG decoder and the model runs on
the eIQ Neutron NPU. On ``native_sim`` the software JPEG decoder is used and the model runs on the
TFLM reference kernels, which allows testing the whole flow on a host.

.. graphviz::

   digraph pipeline {
     rankdir=LR;
     node [shape=box, style=filled, fillcolor="#e8e8e8"];
     file_src     [label="file_src\n(fs)"];
     jpeg_parser  [label="jpeg_parser\n(img)"];
     caps_filter  [label="caps_filter\n(base)"];
     jpegdec      [label="JPEG decoder\n(HW or SW)"];
     app_sink     [label="app_sink\n(base)"];
     infer        [label="inference thread\n(TFLM + Neutron)", fillcolor="#d0e0ff"];
     file_src -> jpeg_parser -> caps_filter -> jpegdec -> app_sink;
     app_sink -> infer [label="frame callback"];
   }

For every decoded frame, the ``app_sink`` callback scales the frame to the model input size
(averaging the source area of each output pixel), converts it to RGB and quantizes it into the
model input tensor, then hands it to the inference thread. The frame buffer is returned to the
decoder before the model runs. By default the decoder waits while the model is busy, so every
frame is classified; :kconfig:option:`CONFIG_INFERENCE_DROP_FRAMES` drops them instead.

The default model is the MLPerf Tiny Visual Wake Words reference model (MobileNetV1 0.25,
96x96 RGB, person / no person), fetched at build time from the `mlcommons/tiny`_ repository.

Requirements
************

- The TensorFlow Lite Micro module:

  .. code-block:: console

     west config manifest.project-filter -- +tflite-micro
     west update tflite-micro

- For the i.MX RT700: the eIQ Neutron SDK (tested with 3.2.3). The build compiles the model with
  its ``neutron-compiler`` and links the Neutron driver and firmware libraries from the same SDK,
  replacing those of ``hal_nxp``: the driver rejects a model compiled by a different compiler
  version.

Model
*****

Any int8 TFLite image classification model taking an NHWC input with 1 or 3 channels, normalized
to [0, 1], and producing int8 class scores can be used:

- Pass it with ``-DMODEL_TFLITE=<file>``. On the RT700 it is compiled for the NPU at build time.
- Set the class names with :kconfig:option:`CONFIG_MODEL_LABELS`.
- Size :kconfig:option:`CONFIG_TENSOR_ARENA_SIZE` from the arena usage logged at start-up.

The operators the Neutron compiler leaves on the CPU must be registered in
``src/inference.cpp``; the compiler output in the build log lists them.

Preparing the SD card
*********************

The input is a single file, :kconfig:option:`CONFIG_FILE_INPUT_PATH` (default
``/SD:/input.mjpeg``), made of baseline 4:2:0 JPEG frames of the size set by
:kconfig:option:`CONFIG_JPEG_IMAGE_WIDTH` and :kconfig:option:`CONFIG_JPEG_IMAGE_HEIGHT` (default
576x432). Concatenate the JPEG files, or convert images or a video with FFmpeg:

.. code-block:: console

   ffmpeg -i frame_%03d.png -vf "scale=576:432,format=yuvj420p" -q:v 3 -f mjpeg input.mjpeg

Copy ``input.mjpeg`` to the root of a FAT-formatted SD card.

Alternatively, pass ``-DTEST_MJPEG=<file>`` to embed an MJPEG file in the image: it is written to
the volume at start-up when the input file is missing.

Building and Running
********************

i.MX RT700 EVK
==============

.. zephyr-app-commands::
   :zephyr-app: samples/subsys/mpipe/jpeg_npu
   :board: mimxrt700_evk/mimxrt798s/cm33_cpu0
   :gen-args: -DNEUTRON_SDK_DIR=<path to eiq-neutron-sdk>
   :goals: build flash

The model and the tensor arena do not fit the 512 KiB default RAM: they are placed in SRAM1
(:kconfig:option:`CONFIG_MODEL_ZEPHYR_REGION`), next to the video buffers, within the SRAM the NPU
can address.

Native simulator
================

.. zephyr-app-commands::
   :zephyr-app: samples/subsys/mpipe/jpeg_npu
   :board: native_sim/native/64
   :gen-args: -DTEST_MJPEG=<file.mjpeg>
   :goals: build run

Sample Output
=============

.. code-block:: console

   <inf> main: SD card mounted at /SD:
   <inf> inference: Model 333288 bytes, input 96x96x3, 2 classes, arena 104240 of 262144 bytes used
   <inf> mpipe_player: Player #0 state: PLAYING
   <inf> main: Frame 0: person 83% (preprocess 0 us, inference 0 us, 0 dropped)
   <inf> main: Frame 1: no person 91% (preprocess 0 us, inference 0 us, 0 dropped)
   <inf> main: Frame 2: no person 97% (preprocess 0 us, inference 0 us, 0 dropped)
   <inf> mpipe_player: Player #0: end of stream

The player shell stays available after the end of the stream: ``r`` replays the file and ``q``
quits.

.. _mlcommons/tiny: https://github.com/mlcommons/tiny
