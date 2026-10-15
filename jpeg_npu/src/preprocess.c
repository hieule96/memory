/*
 * Copyright 2026 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <errno.h>
#include <math.h>

#include <zephyr/drivers/video.h>
#include <zephyr/sys/util.h>

#include "preprocess.h"

/* Samples per axis averaged into one output pixel, bounding the cost on large frames */
#define AREA_SAMPLES 4U

/* Quantized value of each 8-bit pixel level, for the current model input */
static int8_t quant_lut[256];

void preprocess_init(const struct inference_input *input)
{
	__ASSERT_NO_MSG(input != NULL);

	for (int i = 0; i < 256; i++) {
		int32_t q =
			(int32_t)lroundf(((float)i / 255.0f) / input->scale) + input->zero_point;

		quant_lut[i] = (int8_t)CLAMP(q, INT8_MIN, INT8_MAX);
	}
}

static size_t frame_bytes(uint32_t pixfmt, size_t pixels)
{
	switch (pixfmt) {
	case VIDEO_PIX_FMT_NV12:
		return pixels + pixels / 2U;
	case VIDEO_PIX_FMT_YUYV:
	case VIDEO_PIX_FMT_RGB565:
	case VIDEO_PIX_FMT_RGB565X:
		return pixels * 2U;
	case VIDEO_PIX_FMT_YUV24:
	case VIDEO_PIX_FMT_BGR24:
		return pixels * 3U;
	case VIDEO_PIX_FMT_GREY:
		return pixels;
	default:
		return 0;
	}
}

static void yuv_to_rgb(uint8_t y, uint8_t u8, uint8_t v8, uint8_t rgb[3])
{
	int32_t u = (int32_t)u8 - 128;
	int32_t v = (int32_t)v8 - 128;

	rgb[0] = (uint8_t)CLAMP((int32_t)y + 1402 * v / 1000, 0, 255);
	rgb[1] = (uint8_t)CLAMP((int32_t)y - (344 * u + 714 * v) / 1000, 0, 255);
	rgb[2] = (uint8_t)CLAMP((int32_t)y + 1772 * u / 1000, 0, 255);
}

/* RGB value of pixel (x, y); the format was checked by the caller */
static void read_pixel(const struct preprocess_frame *f, uint32_t x, uint32_t y, uint8_t rgb[3])
{
	const uint8_t *d = f->data;
	size_t idx = (size_t)y * f->width + x;
	uint16_t px;

	switch (f->pixfmt) {
	case VIDEO_PIX_FMT_NV12: {
		/* The interleaved UV plane follows the Y plane, one pair per 2x2 block */
		const uint8_t *uv =
			d + (size_t)f->width * f->height + (size_t)(y / 2U) * f->width + (x & ~1U);

		yuv_to_rgb(d[idx], uv[0], uv[1], rgb);
		break;
	}
	case VIDEO_PIX_FMT_YUYV: {
		/* Y0 U Y1 V: one U/V pair per 2 pixels */
		const uint8_t *pair = d + ((size_t)y * f->width + (x & ~1U)) * 2U;

		yuv_to_rgb(pair[(x & 1U) * 2U], pair[1], pair[3], rgb);
		break;
	}
	case VIDEO_PIX_FMT_YUV24:
		yuv_to_rgb(d[idx * 3U], d[idx * 3U + 1U], d[idx * 3U + 2U], rgb);
		break;
	case VIDEO_PIX_FMT_BGR24:
		rgb[0] = d[idx * 3U + 2U];
		rgb[1] = d[idx * 3U + 1U];
		rgb[2] = d[idx * 3U];
		break;
	case VIDEO_PIX_FMT_GREY:
		rgb[0] = d[idx];
		rgb[1] = d[idx];
		rgb[2] = d[idx];
		break;
	case VIDEO_PIX_FMT_RGB565:
	case VIDEO_PIX_FMT_RGB565X:
		px = (f->pixfmt == VIDEO_PIX_FMT_RGB565)
			     ? (uint16_t)(d[idx * 2U] | (d[idx * 2U + 1U] << 8))
			     : (uint16_t)((d[idx * 2U] << 8) | d[idx * 2U + 1U]);
		/* Expand to 8 bits, replicating the high bits into the low ones */
		rgb[0] = (uint8_t)(((px >> 8) & 0xF8U) | ((px >> 13) & 0x07U));
		rgb[1] = (uint8_t)(((px >> 3) & 0xFCU) | ((px >> 9) & 0x03U));
		rgb[2] = (uint8_t)(((px << 3) & 0xF8U) | ((px >> 2) & 0x07U));
		break;
	default:
		break;
	}
}

int preprocess_run(const struct preprocess_frame *frame, const struct inference_input *input)
{
	__ASSERT_NO_MSG(frame != NULL);
	__ASSERT_NO_MSG(input != NULL);

	size_t need = frame_bytes(frame->pixfmt, (size_t)frame->width * frame->height);

	if (need == 0) {
		return -ENOTSUP;
	}

	if (frame->size < need || frame->width == 0 || frame->height == 0) {
		return -EINVAL;
	}

	int8_t *out = input->data;

	for (uint32_t oy = 0; oy < input->height; oy++) {
		uint32_t y0 = (oy * frame->height) / input->height;
		uint32_t y1 = MAX(((oy + 1U) * frame->height) / input->height, y0 + 1U);
		uint32_t ystep = MAX((y1 - y0) / AREA_SAMPLES, 1U);

		for (uint32_t ox = 0; ox < input->width; ox++) {
			uint32_t x0 = (ox * frame->width) / input->width;
			uint32_t x1 = MAX(((ox + 1U) * frame->width) / input->width, x0 + 1U);
			uint32_t xstep = MAX((x1 - x0) / AREA_SAMPLES, 1U);
			uint32_t sum[3] = {0};
			uint32_t n = 0;

			/*
			 * Average the source area an output pixel covers: a single
			 * sample per pixel aliases fine detail into noise at the
			 * usual 4x to 8x reduction, and the model's score with it.
			 */
			for (uint32_t sy = y0; sy < y1; sy += ystep) {
				for (uint32_t sx = x0; sx < x1; sx += xstep) {
					uint8_t rgb[3];

					read_pixel(frame, sx, sy, rgb);
					sum[0] += rgb[0];
					sum[1] += rgb[1];
					sum[2] += rgb[2];
					n++;
				}
			}

			uint8_t r = (uint8_t)((sum[0] + n / 2U) / n);
			uint8_t g = (uint8_t)((sum[1] + n / 2U) / n);
			uint8_t b = (uint8_t)((sum[2] + n / 2U) / n);

			if (input->channels == 1U) {
				/* BT.601 luma */
				*out++ = quant_lut[(299U * r + 587U * g + 114U * b) / 1000U];
			} else {
				*out++ = quant_lut[r];
				*out++ = quant_lut[g];
				*out++ = quant_lut[b];
			}
		}
	}

	return 0;
}
