# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(jpeg_npu)

target_sources(app PRIVATE
	src/main.c
	src/preprocess.c
	src/inference.cpp
)

# int8 TFLite model to deploy. Without one, the MLPerf Tiny Visual Wake Words
# reference model (MobileNetV1 0.25, 96x96 RGB, person / no person) is fetched.
set(MODEL_TFLITE "" CACHE FILEPATH "int8 TFLite model to deploy")
if(NOT MODEL_TFLITE)
	set(MODEL_TFLITE ${CMAKE_BINARY_DIR}/model/vww_96_int8.tflite)
	if(NOT EXISTS ${MODEL_TFLITE})
		file(DOWNLOAD
			https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite
			${MODEL_TFLITE}
			EXPECTED_HASH SHA256=597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b
			STATUS download_status
		)
		list(GET download_status 0 download_error)
		if(NOT download_error EQUAL 0)
			file(REMOVE ${MODEL_TFLITE})
			message(FATAL_ERROR "Downloading the default model failed: ${download_status}. "
				"Pass an int8 TFLite model with -DMODEL_TFLITE=<file>.")
		endif()
	endif()
endif()

if(CONFIG_TFLM_NXP_NEUTRON)
	# The compiler and the runtime libraries come from the same eIQ Neutron SDK:
	# the driver refuses microcode produced by a different compiler version.
	set(NEUTRON_SDK_DIR "$ENV{NEUTRON_SDK_DIR}" CACHE PATH "eIQ Neutron SDK directory")
	set(neutron_compiler ${NEUTRON_SDK_DIR}/bin/neutron-compiler)
	set(neutron_lib_dir ${NEUTRON_SDK_DIR}/target/imxrt700/rt700/cm33)

	if(NOT EXISTS ${neutron_compiler} OR NOT EXISTS ${neutron_lib_dir}/libNeutronDriver.a)
		message(FATAL_ERROR "eIQ Neutron SDK not found in '${NEUTRON_SDK_DIR}'. "
			"Pass it with -DNEUTRON_SDK_DIR=<dir> or the NEUTRON_SDK_DIR environment variable.")
	endif()

	set_target_properties(neutron_driver PROPERTIES
		IMPORTED_LOCATION ${neutron_lib_dir}/libNeutronDriver.a)
	set_target_properties(neutron_firmware PROPERTIES
		IMPORTED_LOCATION ${neutron_lib_dir}/libNeutronFirmware.a)

	# Sequencer mode is faster but needs the model in SRAM, where the
	# application keeps it anyway: the NPU cannot read the flash.
	set(model_bin ${CMAKE_BINARY_DIR}/model/model_npu.tflite)
	add_custom_command(
		OUTPUT ${model_bin}
		COMMAND ${neutron_compiler}
			--input ${MODEL_TFLITE}
			--output ${model_bin}
			--target imxrt700
			--use-sequencer
			--dump-statistics
		DEPENDS ${MODEL_TFLITE}
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/model
		COMMENT "Compiling ${MODEL_TFLITE} for the Neutron NPU"
	)
else()
	set(model_bin ${MODEL_TFLITE})
endif()

generate_inc_file_for_target(app ${model_bin} ${ZEPHYR_BINARY_DIR}/include/generated/model.inc)

# Optional MJPEG file embedded in the image and written to the volume when the
# input file is missing, for boards without a prepared SD card.
set(TEST_MJPEG "" CACHE FILEPATH "MJPEG file to embed as test input")
if(TEST_MJPEG)
	generate_inc_file_for_target(app ${TEST_MJPEG}
		${ZEPHYR_BINARY_DIR}/include/generated/mjpeg.inc)
	target_compile_definitions(app PRIVATE EMBED_TEST_MJPEG)
endif()
