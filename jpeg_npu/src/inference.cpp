/*
 * Copyright 2026 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <errno.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/linker/section_tags.h>
#include <zephyr/logging/log.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#if defined(CONFIG_TFLM_NXP_NEUTRON)
#include <tensorflow/lite/micro/kernels/neutron/neutron.h>
#endif

#include "inference.h"

LOG_MODULE_REGISTER(inference, CONFIG_LOG_DEFAULT_LEVEL);

/*
 * The model is not used from flash: the NPU reads the microcode and weights
 * embedded in it straight from memory, and it reaches the internal SRAM only.
 * The NPU also writes the tensors itself, behind the CPU data cache.
 */
#if defined(CONFIG_MODEL_ZEPHYR_REGION)
#define MODEL_SECTION Z_GENERIC_SECTION(CONFIG_MODEL_ZEPHYR_REGION_NAME)

/* A memory region is not loaded at boot: inference_init() copies the model in */
static const uint8_t model_image[] = {
#include "model.inc"
};
alignas(16) static uint8_t model_data[sizeof(model_image)] MODEL_SECTION;
alignas(16) static uint8_t tensor_arena[CONFIG_TENSOR_ARENA_SIZE] MODEL_SECTION;
#else
alignas(16) static uint8_t model_data[] = {
#include "model.inc"
};
alignas(16) static uint8_t tensor_arena[CONFIG_TENSOR_ARENA_SIZE] __nocache;
#endif

static tflite::MicroInterpreter *interpreter;

static int register_ops(tflite::MicroMutableOpResolver<8> &resolver)
{
	/*
	 * NeutronGraph runs what the Neutron compiler offloaded. The builtins
	 * cover the layers it left on the CPU and the uncompiled model used
	 * without NPU.
	 */
#if defined(CONFIG_TFLM_NXP_NEUTRON)
	if (resolver.AddCustom(tflite::GetString_NEUTRON_GRAPH(),
			       tflite::Register_NEUTRON_GRAPH()) != kTfLiteOk) {
		return -EINVAL;
	}
#endif
	if (resolver.AddAveragePool2D() != kTfLiteOk || resolver.AddConv2D() != kTfLiteOk ||
	    resolver.AddDepthwiseConv2D() != kTfLiteOk ||
	    resolver.AddFullyConnected() != kTfLiteOk || resolver.AddReshape() != kTfLiteOk ||
	    resolver.AddSoftmax() != kTfLiteOk) {
		return -EINVAL;
	}

	return 0;
}

int inference_init(struct inference_input *input, size_t *num_classes)
{
	static tflite::MicroMutableOpResolver<8> resolver;
	int ret;

	__ASSERT_NO_MSG(input != NULL);
	__ASSERT_NO_MSG(num_classes != NULL);

#if defined(CONFIG_MODEL_ZEPHYR_REGION)
	memcpy(model_data, model_image, sizeof(model_data));
#endif

	const tflite::Model *model = tflite::GetModel(model_data);

	if (model->version() != TFLITE_SCHEMA_VERSION) {
		LOG_ERR("Model schema %u, expected %d", (unsigned int)model->version(),
			TFLITE_SCHEMA_VERSION);
		return -EINVAL;
	}

	ret = register_ops(resolver);
	if (ret != 0) {
		LOG_ERR("Failed to register operators (%d)", ret);
		return ret;
	}

	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
							   sizeof(tensor_arena));

	if (static_interpreter.AllocateTensors() != kTfLiteOk) {
		LOG_ERR("Tensor allocation failed, raise CONFIG_TENSOR_ARENA_SIZE");
		return -ENOMEM;
	}

	TfLiteTensor *in = static_interpreter.input(0);
	TfLiteTensor *out = static_interpreter.output(0);

	if (in->type != kTfLiteInt8 || in->dims->size != 4 || in->dims->data[0] != 1 ||
	    (in->dims->data[3] != 1 && in->dims->data[3] != 3)) {
		LOG_ERR("Input must be int8 [1, H, W, 1 or 3]");
		return -EINVAL;
	}

	if (out->type != kTfLiteInt8 || out->dims->size != 2) {
		LOG_ERR("Output must be int8 [1, classes]");
		return -EINVAL;
	}

	input->data = in->data.int8;
	input->height = (uint16_t)in->dims->data[1];
	input->width = (uint16_t)in->dims->data[2];
	input->channels = (uint16_t)in->dims->data[3];
	input->scale = in->params.scale;
	input->zero_point = in->params.zero_point;
	*num_classes = (size_t)out->dims->data[1];

	interpreter = &static_interpreter;

	LOG_INF("Model %u bytes, input %ux%ux%u, %u classes, arena %u of %u bytes used",
		(unsigned int)sizeof(model_data), input->width, input->height, input->channels,
		(unsigned int)*num_classes, (unsigned int)static_interpreter.arena_used_bytes(),
		(unsigned int)sizeof(tensor_arena));

	return 0;
}

int inference_run(float *scores, size_t num_scores, uint32_t *usec)
{
	__ASSERT_NO_MSG(interpreter != NULL);
	__ASSERT_NO_MSG(scores != NULL);
	__ASSERT_NO_MSG(usec != NULL);

	uint32_t start = k_cycle_get_32();

	if (interpreter->Invoke() != kTfLiteOk) {
		return -EIO;
	}

	*usec = (uint32_t)k_cyc_to_us_floor64(k_cycle_get_32() - start);

	TfLiteTensor *out = interpreter->output(0);
	size_t n = MIN(num_scores, (size_t)out->dims->data[1]);

	for (size_t i = 0; i < n; i++) {
		scores[i] = out->params.scale * (float)(out->data.int8[i] - out->params.zero_point);
	}

	return 0;
}
