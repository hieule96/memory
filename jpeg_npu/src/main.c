/*
 * Copyright 2026 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <errno.h>
#include <string.h>
#if defined(CONFIG_FAT_FILESYSTEM_ELM)
#include <ff.h>
#endif

#include <zephyr/drivers/video.h>
#include <zephyr/fs/fs.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <zephyr/mpipe/mpipe.h>
#include <zephyr/mpipe/base/mpipe_app_sink.h>
#include <zephyr/mpipe/base/mpipe_caps_filter.h>
#include <zephyr/mpipe/fs/mpipe_file_src.h>
#include <zephyr/mpipe/img/mpipe_img_jpeg_decoder.h>
#include <zephyr/mpipe/img/mpipe_img_jpeg_parser.h>
#include <zephyr/mpipe/utils/mpipe_player.h>
#if DT_HAS_CHOSEN(zephyr_jpegdec)
#include <zephyr/mpipe/vid/mpipe_vid_transform.h>
#endif

#include "inference.h"
#include "preprocess.h"

LOG_MODULE_REGISTER(main, CONFIG_LOG_DEFAULT_LEVEL);

/* Element IDs (values are arbitrary; only uniqueness within the pipeline matters) */
enum {
	PIPE_ID,
	FILE_SRC_ID,
	JPEG_PARSER_ID,
	CAPS_FILTER_ID,
	JPEG_DEC_ID,
	APP_SINK_ID,
};

#define MNT_POINT   "/SD:"
#define MAX_CLASSES 32

#if defined(CONFIG_FAT_FILESYSTEM_ELM)
static FATFS fat_fs;
#endif

static struct fs_mount_t mp = {
	.type = FS_FATFS,
#if defined(CONFIG_FAT_FILESYSTEM_ELM)
	.fs_data = &fat_fs,
#endif
	.mnt_point = MNT_POINT,
};

static struct mpipe pipe;
static struct mpipe_file_src file_src;
static struct mpipe_img_jpeg_parser jpeg_parser;
static struct mpipe_caps_filter caps_filter;
#if DT_HAS_CHOSEN(zephyr_jpegdec)
static struct mpipe_vid_transform jpeg_dec;
#else
static struct mpipe_img_jpeg_decoder jpeg_dec;
#endif
static struct mpipe_app_sink app_sink;
static struct mpipe_player player;

static struct inference_input model_in;
static size_t num_classes;
static const char *labels[MAX_CLASSES];
static char labels_buf[sizeof(CONFIG_MODEL_LABELS)];

K_THREAD_STACK_DEFINE(inference_stack, CONFIG_INFERENCE_STACK_SIZE);
static struct k_thread inference_tid;

#if defined(EMBED_TEST_MJPEG)
static const uint8_t mjpeg[] = {
#include "mjpeg.inc"
};

/* Write the embedded test input unless the volume already holds an input file */
static int embed_test_file(void)
{
	struct fs_dirent ent;
	struct fs_file_t f;
	ssize_t w;
	int ret = fs_stat(CONFIG_FILE_INPUT_PATH, &ent);

	if (ret == 0) {
		return 0;
	}

	if (ret != -ENOENT) {
		LOG_ERR("fs_stat(%s) failed (%d)", CONFIG_FILE_INPUT_PATH, ret);
		return ret;
	}

	fs_file_t_init(&f);

	ret = fs_open(&f, CONFIG_FILE_INPUT_PATH, FS_O_CREATE | FS_O_WRITE);
	if (ret != 0) {
		LOG_ERR("fs_open(%s) failed (%d)", CONFIG_FILE_INPUT_PATH, ret);
		return ret;
	}

	w = fs_write(&f, mjpeg, sizeof(mjpeg));
	(void)fs_close(&f);

	if (w < 0 || (size_t)w != sizeof(mjpeg)) {
		LOG_ERR("fs_write failed (%d)", (int)w);
		return (w < 0) ? (int)w : -EIO;
	}

	LOG_INF("Wrote embedded test input to %s (%u bytes)", CONFIG_FILE_INPUT_PATH,
		(unsigned int)sizeof(mjpeg));

	return 0;
}
#endif

static int mount_sd(void)
{
	int ret = fs_mount(&mp);

	if (ret != 0) {
		LOG_ERR("fs_mount failed (%d)", ret);
		return ret;
	}

	LOG_INF("SD card mounted at %s", MNT_POINT);

#if defined(EMBED_TEST_MJPEG)
	return embed_test_file();
#else
	return 0;
#endif
}

/* Split CONFIG_MODEL_LABELS at the commas; missing labels print as "?" */
static void parse_labels(void)
{
	size_t n = 0;
	char *s = labels_buf;

	memcpy(labels_buf, CONFIG_MODEL_LABELS, sizeof(labels_buf));

	while (n < ARRAY_SIZE(labels)) {
		labels[n++] = s;
		s = strchr(s, ',');
		if (s == NULL) {
			break;
		}
		*s++ = '\0';
	}

	for (; n < ARRAY_SIZE(labels); n++) {
		labels[n] = "?";
	}
}

/* Format and size the pipeline negotiated for the frames reaching the sink */
static int sink_frame_format(struct preprocess_frame *frame)
{
	const struct mpipe_structure *caps = &app_sink.sink.sink_pad.caps;
	const struct mpipe_value *fmt = mpipe_structure_get_value(caps, MPIPE_CAPS_PIXEL_FORMAT);
	const struct mpipe_value *w = mpipe_structure_get_value(caps, MPIPE_CAPS_IMAGE_WIDTH);
	const struct mpipe_value *h = mpipe_structure_get_value(caps, MPIPE_CAPS_IMAGE_HEIGHT);

	if (fmt == NULL || w == NULL || h == NULL) {
		return -EINVAL;
	}

	frame->pixfmt = mpipe_value_get_uint(fmt);
	frame->width = (uint16_t)mpipe_value_get_uint(w);
	frame->height = (uint16_t)mpipe_value_get_uint(h);

	return 0;
}

/*
 * Handshake between the streaming thread, which fills the input tensor, and
 * the inference thread, which runs the model on it. The tensor is in use
 * while a frame is being classified: the next frame waits for it, or is
 * dropped with CONFIG_INFERENCE_DROP_FRAMES.
 */
static K_SEM_DEFINE(model_idle, 1, 1);
static K_SEM_DEFINE(frame_ready, 0, 1);
static uint32_t frame_num;
static uint32_t frame_pre_us;
static uint32_t frames_dropped;

/*
 * Runs on the streaming thread for every decoded frame. Converting the frame
 * here, rather than queueing it for the inference thread, returns the decoder's
 * buffer at once: holding it would starve decoders that own a single output
 * buffer.
 */
static void frame_cb(const struct net_buf *buf, void *user_data)
{
	ARG_UNUSED(user_data);

	struct preprocess_frame frame = {
		.data = buf->data,
		.size = mpipe_buffer_get_meta(buf)->bytes_used,
	};
	uint32_t start = k_cycle_get_32();
	int ret;

	if (k_sem_take(&model_idle,
		       IS_ENABLED(CONFIG_INFERENCE_DROP_FRAMES) ? K_NO_WAIT : K_FOREVER) != 0) {
		frames_dropped++;
		return;
	}

	ret = sink_frame_format(&frame);
	if (ret == 0) {
		ret = preprocess_run(&frame, &model_in);
	}

	if (ret != 0) {
		LOG_ERR("Frame %u: preprocessing failed (%d), format %c%c%c%c %ux%u", frame_num,
			ret, (char)frame.pixfmt, (char)(frame.pixfmt >> 8),
			(char)(frame.pixfmt >> 16), (char)(frame.pixfmt >> 24), frame.width,
			frame.height);
		frame_num++;
		k_sem_give(&model_idle);
		return;
	}

	frame_pre_us = (uint32_t)k_cyc_to_us_floor64(k_cycle_get_32() - start);
	k_sem_give(&frame_ready);
}

static void inference_thread(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	while (true) {
		float scores[MAX_CLASSES];
		uint32_t run_us;
		size_t best = 0;
		int ret;

		(void)k_sem_take(&frame_ready, K_FOREVER);

		ret = inference_run(scores, num_classes, &run_us);
		if (ret != 0) {
			LOG_ERR("Frame %u: inference failed (%d)", frame_num, ret);
		} else {
			for (size_t i = 1; i < num_classes; i++) {
				if (scores[i] > scores[best]) {
					best = i;
				}
			}

			LOG_INF("Frame %u: %s %u%% (preprocess %u us, inference %u us, "
				"%u dropped)",
				frame_num, labels[best],
				(unsigned int)(scores[best] * 100.0f + 0.5f), frame_pre_us, run_us,
				frames_dropped);
		}

		frame_num++;
		k_sem_give(&model_idle);
	}
}

static int setup_pipeline(void)
{
	int ret = mpipe_pipeline_init(&pipe, PIPE_ID);

	if (ret != 0) {
		return ret;
	}

	ret = mpipe_file_src_init(&file_src, FILE_SRC_ID);
	if (ret != 0) {
		return ret;
	}

	ret = mpipe_img_jpeg_parser_init(&jpeg_parser, JPEG_PARSER_ID);
	if (ret != 0) {
		return ret;
	}

	ret = mpipe_caps_filter_init(&caps_filter, CAPS_FILTER_ID);
	if (ret != 0) {
		return ret;
	}

#if DT_HAS_CHOSEN(zephyr_jpegdec)
	ret = mpipe_vid_transform_init(&jpeg_dec, JPEG_DEC_ID);
	if (ret != 0) {
		return ret;
	}

	ret = mpipe_object_set_properties((struct mpipe_object *)&jpeg_dec, MPIPE_PROP_VID_DEVICE,
					  DEVICE_DT_GET(DT_CHOSEN(zephyr_jpegdec)),
					  MPIPE_PROP_LIST_END);
	if (ret != 0) {
		return ret;
	}
#else
	ret = mpipe_img_jpeg_decoder_init(&jpeg_dec, JPEG_DEC_ID);
	if (ret != 0) {
		return ret;
	}
#endif

	ret = mpipe_app_sink_init(&app_sink, APP_SINK_ID);
	if (ret != 0) {
		return ret;
	}

	static const struct mpipe_app_sink_cb cb = {.fn = frame_cb};

	ret = mpipe_object_set_properties((struct mpipe_object *)&app_sink,
					  MPIPE_PROP_BASE_APP_SINK_CB, &cb, MPIPE_PROP_LIST_END);
	if (ret != 0) {
		return ret;
	}

	ret = mpipe_object_set_properties((struct mpipe_object *)&file_src, MPIPE_PROP_FS_SRC_PATH,
					  CONFIG_FILE_INPUT_PATH, MPIPE_PROP_LIST_END);
	if (ret != 0) {
		return ret;
	}

	struct mpipe_structure caps;

	ret = mpipe_structure_init_fields(
		&caps, MPIPE_MEDIA_VIDEO, MPIPE_CAPS_PIXEL_FORMAT, MPIPE_TYPE_UINT,
		VIDEO_PIX_FMT_JPEG, MPIPE_CAPS_IMAGE_WIDTH, MPIPE_TYPE_UINT,
		CONFIG_JPEG_IMAGE_WIDTH, MPIPE_CAPS_IMAGE_HEIGHT, MPIPE_TYPE_UINT,
		CONFIG_JPEG_IMAGE_HEIGHT, MPIPE_CAPS_END);
	if (ret != 0) {
		return ret;
	}

	ret = mpipe_object_set_properties((struct mpipe_object *)&caps_filter,
					  MPIPE_PROP_BASE_CAPS_FILTER_CAPS, &caps,
					  MPIPE_PROP_LIST_END);
	if (ret != 0) {
		return ret;
	}

	ret = mpipe_bin_add((struct mpipe_bin *)&pipe, (struct mpipe_element *)&file_src,
			    (struct mpipe_element *)&jpeg_parser,
			    (struct mpipe_element *)&caps_filter, (struct mpipe_element *)&jpeg_dec,
			    (struct mpipe_element *)&app_sink, NULL);
	if (ret != 0) {
		LOG_ERR("Failed to add elements (%d)", ret);
		return ret;
	}

	ret = mpipe_element_link(
		(struct mpipe_element *)&file_src, (struct mpipe_element *)&jpeg_parser,
		(struct mpipe_element *)&caps_filter, (struct mpipe_element *)&jpeg_dec,
		(struct mpipe_element *)&app_sink, NULL);
	if (ret != 0) {
		LOG_ERR("Failed to link elements (%d)", ret);
		return ret;
	}

	return 0;
}

int main(void)
{
	int ret = mount_sd();

	if (ret != 0) {
		return 0;
	}

	ret = inference_init(&model_in, &num_classes);
	if (ret != 0) {
		LOG_ERR("Model setup failed (%d)", ret);
		return 0;
	}

	if (num_classes > MAX_CLASSES) {
		LOG_WRN("Model has %u classes, only the first %u are scored",
			(unsigned int)num_classes, MAX_CLASSES);
		num_classes = MAX_CLASSES;
	}

	preprocess_init(&model_in);
	parse_labels();

	ret = setup_pipeline();
	if (ret != 0) {
		LOG_ERR("Pipeline setup failed (%d)", ret);
		return 0;
	}

	(void)k_thread_create(&inference_tid, inference_stack,
			      K_THREAD_STACK_SIZEOF(inference_stack), inference_thread, NULL, NULL,
			      NULL, CONFIG_INFERENCE_PRIORITY, 0, K_NO_WAIT);
	(void)k_thread_name_set(&inference_tid, "inference");

	ret = mpipe_player_init(&player, &pipe);
	if (ret != 0) {
		LOG_ERR("Failed to init player (%d)", ret);
		return 0;
	}

	(void)mpipe_player_play(&player);
	(void)mpipe_player_wait_quit(&player);
	(void)mpipe_player_deinit(&player);

	ret = fs_unmount(&mp);
	if (ret != 0) {
		LOG_ERR("fs_unmount failed (%d)", ret);
	}

	LOG_INF("Done.");

	return 0;
}
