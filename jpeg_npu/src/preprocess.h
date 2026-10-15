/*
 * Copyright 2026 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef JPEG_NPU_PREPROCESS_H_
#define JPEG_NPU_PREPROCESS_H_

#include <stddef.h>
#include <stdint.h>

#include "inference.h"

/** A decoded frame as it leaves the pipeline */
struct preprocess_frame {
	/** Pixel data, planes back to back with no row padding */
	const uint8_t *data;
	/** Valid bytes in @p data */
	size_t size;
	/** VIDEO_PIX_FMT_* code */
	uint32_t pixfmt;
	/** Frame width in pixels */
	uint16_t width;
	/** Frame height in pixels */
	uint16_t height;
};

/**
 * @brief Prepare the quantization table for a model input
 *
 * The model is assumed to take pixel values normalized to [0, 1].
 *
 * @param input Model input tensor description
 */
void preprocess_init(const struct inference_input *input);

/**
 * @brief Scale a decoded frame to the model input and quantize it
 *
 * Each output pixel averages up to 4x4 samples of the source area it covers.
 * YUV is converted to RGB with the BT.601 full-range
 * (JFIF) equations. Supported formats: NV12, YUYV, YUV24, GREY, BGR24,
 * RGB565 and RGB565X.
 *
 * @param frame The decoded frame
 * @param input Model input tensor to fill
 *
 * @retval 0 Success
 * @retval -ENOTSUP Unsupported pixel format
 * @retval -EINVAL Frame smaller than its format and size require
 */
int preprocess_run(const struct preprocess_frame *frame, const struct inference_input *input);

#endif /* JPEG_NPU_PREPROCESS_H_ */
