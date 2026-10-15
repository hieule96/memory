<!--
SPDX-FileCopyrightText: Copyright The Zephyr Project Contributors
SPDX-License-Identifier: Apache-2.0
-->

# mpipe: Multimedia Pipeline subsystem

Notes for coding agents working on `subsys/mpipe`. The repository-wide rules in the top-level
`AGENTS.md` still apply; this file adds what is specific to mpipe. The user-facing description
is `doc/services/mpipe/index.rst`, and the design rationale lives in the Doxygen blocks of
`include/zephyr/mpipe/*.h`. Read the header of any type before changing its behavior: the
headers state contracts that the code alone does not show.

mpipe is a GStreamer-like framework. Elements (sources, transforms, parsers, sinks) are linked
by pads into a graph held by a pipeline. The pipeline drives the graph through a
READY/PAUSED/PLAYING state machine, negotiates the data format and the buffer pools, and pushes
`net_buf` buffers from the source to the sinks. It is `EXPERIMENTAL` (`@since 4.5`).

## Where things are

| Path | Contents |
| :--- | :--- |
| `include/zephyr/mpipe/mpipe.h` | Umbrella header for the core API, plus the top-level Doxygen groups |
| `include/zephyr/mpipe/*.h` | Core types: object, element, pad, bin, pipeline, src, sink, transform, parser, buffer, structure, value, dispatch, message, thread, workqueue |
| `subsys/mpipe/*.c` | Core implementation, one file per header |
| `subsys/mpipe/<plugin>/` | Plugins: `base`, `vid`, `disp`, `img`, `fs`, `aud`, plus `utils` |
| `include/zephyr/mpipe/<plugin>/` | Plugin headers; `<plugin>/mpipe_<plugin>.h` declares the plugin's Doxygen group |
| `samples/subsys/mpipe/` | `cam_disp`, `jpeg_dec`, `tee_dec`, `dual_pipeline`, `dmic_i2s`, `fs` |
| `tests/subsys/mpipe/` | `unit`, `pipeline`, `base`, `utils`, `build_all` |
| `doc/services/mpipe/index.rst` | Subsystem documentation |

`samples/subsys/mpipe/screen_mirror/` contains only an ignored `build/` directory; it is not a
sample.

### Plugins

| Dir | Kconfig | Elements |
| :--- | :--- | :--- |
| `base` | `MPIPE_BASE` + one option per element | `caps_filter`, `queue` (own thread, leak policy), `tee`, `app_src`, `app_sink` |
| `vid` | `MPIPE_VID` (`MPIPE_VID_CONVERT`) | `vid_src` (camera), `vid_transform` (m2m HW: scaler, rotator, JPEG decoder), `vid_convert` (SW pixel format), `vid_buffer_pool`; RPC clients with `MPIPE_RPC` |
| `disp` | `MPIPE_DISP` | `disp_sink`; caps enumerated from the panel |
| `img` | `MPIPE_IMG` | `jpeg_parser` (`mpipe_parser`), `jpeg_decoder` (SW) |
| `fs` | `MPIPE_FS` | `file_src`, `file_sink`; carry no caps, a parser downstream finds the format |
| `aud` | `MPIPE_AUD` | `dmic_src`, `gain`, `i2s_codec_sink`, `aud_buffer_pool` |
| `utils` | `MPIPE_DUMP`, `MPIPE_PLAYER` | Graphviz DOT dump; player (play/pause/stop/replay/quit, shell commands) |

`img/jpeg_dec.c` is a port of Larry Bank's JPEGDEC (third-party code, BitBank copyright). Do not
restyle it; touch only what a fix needs.

## Build integration

- `subsys/mpipe/CMakeLists.txt` and `Kconfig` auto-discover plugin directories (`file(GLOB)` and
  `osource "subsys/mpipe/*/Kconfig"`). A new plugin adds its own directory with a
  `CMakeLists.txt` guarded by its own `CONFIG_MPIPE_*` and a `Kconfig`; do not edit the top-level
  files for it. Everything compiles into the single `mpipe` zephyr library.
- `MPIPE` selects `NET_BUF`, `ZBUS`, `ZBUS_RUNTIME_OBSERVERS` and
  `ZBUS_RUNTIME_CHANNEL_REGISTRATION`, and forces static zbus allocation. `mpipe_bin.c` has an
  `#error` if `CONFIG_ZBUS_PREFER_DYNAMIC_ALLOCATION` ends up enabled.
- Logging uses one module, `CONFIG_MPIPE_LOG_LEVEL`, with a `LOG_MODULE_REGISTER` per file.

## Object model

Inheritance is C struct embedding: every type embeds its base as the **first member**, so an
upcast is a plain cast (`(struct mpipe_element *)&vid_src`). The chain is

    mpipe_object -> mpipe_element -> {mpipe_src, mpipe_sink, mpipe_transform, mpipe_parser, mpipe_bin}
                                                                                  mpipe_bin -> mpipe (pipeline)
    mpipe_object -> mpipe_pad

A concrete element embeds one base, calls the base init (`mpipe_src_init()`, ...) with its id,
calls `mpipe_element_set_name()` with its type name, then overrides the hooks it needs:
`change_state`, `object.set_property`/`get_property`, the base's `set_caps`,
`transform_caps`, `propose_buffer_pool`, `decide_buffer_pool`, and the pad's `chain_fn`,
`query_fn`, `event_fn`, `enum_caps_fn`. `fs/mpipe_file_src.c` is a short, complete example.

### Properties

Properties configure one element (`mpipe_object_set_properties(obj, key, val, ...,
MPIPE_PROP_LIST_END)`). Keys are layered enums: each base ends its enum with a `*_LAST`
marker (`MPIPE_PROP_SRC_LAST`, `MPIPE_PROP_SINK_LAST`, `MPIPE_PROP_TRANSFORM_LAST`,
`MPIPE_PROP_PARSER_LAST`), and a derived element's enum starts at its base's marker. The derived
setter/getter handles its own keys and delegates `default:` to the base
(`mpipe_src_set_property()` etc.). Video elements also pass `VIDEO_CID_*` keys straight to the
driver's controls (`vid/mpipe_vid_object.c`).

A property is not a capability: a capability field is something two linked elements must agree
on (see the `mpipe_caps_field` comment in `mpipe_structure.h` for the test).

## Runtime model

### States

`READY <-> PAUSED <-> PLAYING`, one step at a time (`MPIPE_STATE_GET_NEXT`). A bin sorts its
children topologically and transitions them sink-to-source going up, source-to-sink going down.
A refused transition is not unwound; the element stays where it is.

- **READY -> PAUSED**: the source drives caps negotiation, then the buffer pool query; pools
  start. The pipeline creates its streaming thread (not started) and counts its sinks.
- **PAUSED -> PLAYING**: the pipeline thread is resumed.
- **PLAYING -> PAUSED**: the thread is paused only. Queued buffers survive; no flushing.
- **PAUSED -> READY**: the pipeline raises the `flushing` gate on every pad *before* the children
  tear down, then joins its thread *after* them (a queue draining unblocks a thread stuck in a
  full `k_msgq_put`). Pads are reset to ANY caps and pools are stopped, so the next run
  renegotiates from scratch.

### Caps negotiation (`mpipe_src.c`, `mpipe_transform.c`, `mpipe_pad.c`)

- A capability is an `mpipe_structure`: media type plus up to
  `CONFIG_MPIPE_STRUCTURE_MAX_FIELDS` (field id, `mpipe_value`) pairs. Values are scalars or
  `[min, max, step]` ranges. Intersection keeps fields only one side carries, which is how
  constraints travel through elements that do not care about them.
- There is no list type. Alternatives live on an **index**: `enum_caps_fn` yields one caps per
  index, `transform_caps` one transformation per index. Return `-EAGAIN` for "this index does
  not match, try the next" and `-ENOENT` for "past the end". An `enum_caps_fn` that never returns
  `-ENOENT` fails at `UINT16_MAX` instead of hanging.
- The source offers each of its caps downstream as a CAPS query; a transform nests two walks
  (its caps x its transformations) and asks its own peer. The first accepted candidate is kept,
  fixated, and sent downstream as a CAPS **event**. A transform forwards the event **before**
  applying caps to itself, because `set_caps` may rewire the graph.
- A parser turns a NULL-caps event (from a format-less source such as `file_src`) into a real
  announcement with the caps it detected.

### Dispatches vs messages

- `mpipe_dispatch` crosses a pad link: CAPS (query or event), BUFFER_POOL (query), EOS (event).
  Query vs event is decided by the function it is passed to (`mpipe_pad_query()` /
  `mpipe_pad_send_event()`), not by a field. The creator owns the `caps` storage; others write
  into it and must never repoint it at their own stack.
- `mpipe_message` goes up the bin's zbus channel to the application (EOS, ERROR with an
  `mpipe_error_domain` and a negative errno). Post with `mpipe_message_post()` only for failures
  fatal to the stream. The pipeline's validator folds per-sink EOS into one.

### Buffers and pools

- Buffers are `net_buf` with `struct mpipe_buffer_meta` in user data
  (`mpipe_buffer_get_meta()`): owning pool, `bytes_used`, timestamp, `driver_buf` for zero-copy.
  Writers set both `bytes_used` and `buf->len`; readers use `bytes_used`.
- `mpipe_buffer_pool` is a vtable (`configure`, `set_config`, `start`, `stop`,
  `acquire_buffer`, `release_buffer`). The BUFFER_POOL query goes downstream to the sink and
  proposals come back upstream; each element decides with its downstream's proposal in hand.
  A proposal is either a whole pool (`query->pool`) or a bare config (`query->pool_cfg`).
- `req_config` is the floor set once at init with `mpipe_buffer_pool_set_req_config()`;
  `config` is the negotiated value, restored from `req_config` by `mpipe_buffer_pool_stop()`.
  In `decide_buffer_pool`/`propose_buffer_pool`, rebuild your own pool's baseline every run,
  otherwise demands accumulate across replays.
- A demand reaches a pool you do not own **only** through `mpipe_buffer_pool_set_config()`,
  which the owner may clamp or refuse; keep a fallback. Whoever starts a pool stops it.
- A source does its real work inside its pool's `acquire_buffer` (read a file chunk, dequeue a
  camera frame). Returning `-ENODATA` means end of stream; `-EPIPE` means a forced stop and is
  not reported as an error.

### Threads

- The pipeline thread finds the first child with no sink pads, loops `acquire_buffer` then
  `mpipe_push_buffer()`, which calls each sink pad's `chain_fn` in turn. A `chain_fn` setting
  `*out_buf = NULL` consumes the buffer and ends the walk (sinks, queue).
- A `chain_fn` **owns** its input buffer: on success or failure it must release or pass it on,
  and the caller never touches it again.
- Thread stacks come from a fixed pool (`mpipe_thread.c`). `CONFIG_MPIPE_THREADS_NUM` must cover
  every concurrently running pipeline plus every `queue` element. Too few gives `-EAGAIN` on
  READY -> PAUSED.
- Negotiation runs on the thread calling `mpipe_element_set_state()` (the player worker, when
  `mpipe_player` is used), not on the streaming thread. It uses roughly 800 bytes of stack per
  element; size `CONFIG_MPIPE_PLAYER_WORKER_STACK_SIZE` or the caller's stack for it.
- `CONFIG_MPIPE_WORKQUEUE` provides a shared P4WQ (`mpipe_p4wq`) for elements that offload work.

## Rules specific to this subsystem

- **No dynamic allocation.** Every type on the negotiation path is fixed-size and held by value;
  buffers come from static `NET_BUF_POOL_FIXED_DEFINE` pools sized by Kconfig. Reintroducing
  allocation on a negotiation path is a bug, however clean it looks. Scratch memory comes from
  the application at init.
- **NULL pointers are asserted, not returned as errors** (`__ASSERT_NO_MSG`), unless the
  parameter's documentation says what NULL means. Bad *values* (empty caps, out-of-range
  property) return a negative errno.
- An override of `change_state` must chain to the base's (`mpipe_src_change_state()`,
  `mpipe_transform_change_state()`, ...); the base performs the pad caps reset and pool teardown.
- A new `mpipe_caps_field` goes before `MPIPE_CAPS_END`, documents its unit and value type, and
  gets a name in `dump_field_names` in `utils/mpipe_dump.c` (a `BUILD_ASSERT` enforces it). Add
  one only if more than one element must agree on it; otherwise it is a property.
- A failure fatal to the stream on the streaming thread has no caller to return to: log it with
  `LOG_ERR` at the site that detected it *and* post an `MPIPE_MESSAGE_ERROR` with the matching
  `mpipe_error_domain`. A failure a caller can see stays a returned errno, without a message.
- Commit prefix is `mpipe:` plus the plugin or area: `mpipe: base:`, `mpipe: img: jpeg_parser:`,
  `mpipe: pad:`, `samples: mpipe: <sample>:`. Check `git log --format=%s -20 -- <path>`.

## Code style

The Zephyr rules in `AGENTS.md` and `doc/contribute/style/code.rst` apply. On top of them, the
mpipe code follows these conventions; match them in new code.

### Files and headers

- A `.c` file starts with the SPDX block (`Copyright <year[-year]> NXP` for existing NXP files;
  extend the year range when you modify one), then includes, then
  `LOG_MODULE_REGISTER(mpipe_<name>, CONFIG_MPIPE_LOG_LEVEL);`.
- Include order: Zephyr and libc headers, a blank line, the core `<zephyr/mpipe/mpipe_*.h>`
  headers sorted, then the plugin's own `<zephyr/mpipe/<plugin>/...>` header. Include what the
  file uses rather than the `mpipe.h` umbrella, which is for applications.
- Header guards are `ZEPHYR_INCLUDE_MPIPE_<SUBDIR>_<FILE>_H_`. Forward-declare a struct
  (`struct mpipe_dispatch;`) instead of including a header only for a pointer type.
- One element per `.c`/`.h` pair, named `mpipe_<plugin>_<element>` (`mpipe_aud_gain.c`); the
  `base` and `fs` elements drop the plugin part (`mpipe_tee.c`, `mpipe_file_src.c`).

### Naming

- Public functions: `mpipe_<element>_<verb>` (`mpipe_queue_init()`, `mpipe_app_src_push()`).
- Static functions use the same prefix. Pad callbacks end in `_fn` and name the pad side
  (`mpipe_tee_sink_query_fn`, `mpipe_tee_sink_event_fn`); pool callbacks read
  `mpipe_<element>_pool_<op>` (`mpipe_file_src_pool_acquire_buffer`).
- The element pointer inside a method is `self`; a typed alias for a derived element uses a
  short name of its type (`fsrc`, `tee`, `src`). The return-code variable is `ret`.
- Property keys `MPIPE_PROP_<PLUGIN>_<ELEMENT>_<NAME>` (`MPIPE_PROP_FS_SRC_PATH`), or
  `MPIPE_PROP_<PLUGIN>_<NAME>` when the key is shared by the plugin's elements
  (`MPIPE_PROP_VID_DEVICE`). Kconfig symbols follow the same shape:
  `CONFIG_MPIPE_BASE_QUEUE_MAX_SIZE`, `CONFIG_MPIPE_FS_BLOCK_SIZE`.

### Statements

- Return values are checked as `if (ret != 0)` (the dominant form; `ret < 0` exists in older
  code). Functions return 0 on success and a negative errno, never a positive value.
- Ignored return values are cast: `(void)mpipe_message_post(&msg);`.
- Pointer parameters a caller must supply are checked with `__ASSERT_NO_MSG(p != NULL)` at the
  top of public functions; there is no other assert form in this subsystem.
- Reach the owning element from a pad with a cast of `pad->object.container`; reach a struct
  from an embedded member (a pool inside an element) with `CONTAINER_OF`.
- Declarations may sit where the value is first known (C99), including inside loops. A `case`
  that declares variables gets its own braces (`case MPIPE_DISPATCH_CAPS: { ... }`), as in
  `base/mpipe_tee.c`.
- Prefer early returns. `goto err` is used only for unwinding in `vid/mpipe_vid_convert.c` and
  `img/mpipe_img_jpeg_parser.c`.
- Messages and dispatches are stack values built with designated initializers; zero is the
  neutral value of every field, so leave unused fields out.
- `#if defined(CONFIG_...)` is kept for struct members that exist only with an option (`name`
  under `CONFIG_MPIPE_DUMP`), `CONFIG_THREAD_NAME`, and shell/dump hooks in the player. Use
  `IS_ENABLED()` for new code paths.
- `/* clang-format off */` is reserved for designated-initializer tables and shell command sets
  (`utils/mpipe_dump.c`, `utils/mpipe_player.c`) and for application property lists built with
  `COND_CODE_0`/`IF_ENABLED` in samples. Do not add it elsewhere.
- Log messages start with a capital letter, have no trailing period, and append the errno as
  `(%d)`: `LOG_ERR("Failed to send EOS event downstream (%d)", ret);`. When a Kconfig option is
  the fix, name it: `"No slot left for field %u, raise CONFIG_MPIPE_STRUCTURE_MAX_FIELDS"`.

## Comment style

Comments in mpipe explain **why** and state **contracts**; the code says what. Maintainers
reject comments that narrate the code and accept ones that stop a question from being reopened
(see commit 9bb38f70a5a, "mpipe: base: Say why the queue is a msgq").

### In `.c` files

- `/* */` only. A one-line comment is a sentence fragment with no trailing period:
  `/* Combine all downstream branch's pool config proposals */`.
- A multi-line comment opens with `/*` alone, continues with ` * `, closes with ` */`, and is
  written in full sentences with periods.
- Put the comment on the line(s) above the code it explains, not at the end of a line.
- Say what would break otherwise: ordering constraints ("raise the flushing gate BEFORE the
  children dismantle their pools"), ownership ("the peer answers by writing into it, so the
  candidate is copied rather than referenced"), and why an obvious alternative was not used.
- Refer to functions as `name()` and to fields by name, so the reference stays greppable.
- A `TODO:` states the missing piece concretely (`/* TODO: Add support for other buffer types */`)
  and is not added for work already done. Remove a TODO when its condition no longer holds.

### Doxygen in headers

- File block: `@file`, `@brief <Type>: <what it is>.` (`@brief Pad: the point where two
  elements meet.`), `@ingroup <group>`.
- Each core header defines one `@defgroup mpipe_<x>` under `mpipe_framework`, opening with a
  narrative that explains the concept, its contracts and the reasoning behind them, with
  `@section` headings for distinct topics. Plugin groups are defined once in
  `include/zephyr/mpipe/<plugin>/mpipe_<plugin>.h` under `mpipe_plugins`, and each element
  header joins with `@ingroup`. Close the group with `@{` ... `@}`.
- Every struct member and enumerator carries `/** ... */` above it (or `/**< ... */` on the same
  line for short enum tables). Function-pointer hooks document ownership and the errnos they may
  return.
- Functions: `@brief`, a body paragraph when the contract is not obvious, `@param` for each
  parameter (stating when NULL is allowed and what it means), `@param[out]` for output storage,
  and `@retval <errno> <condition>` for each discrete value, `0` first.
- Use `@ref` to link types, groups and functions, `@kconfig{CONFIG_...}` for Kconfig symbols,
  `@code{.c}` ... `@endcode` for examples, and `@cond INTERNAL_HIDDEN` ... `@endcond` around
  members applications must not touch.
- Write for the reader of the API: no history, no "now", no references to earlier designs.

### Kconfig help

- First paragraph: what the option does, in one or two sentences. Further paragraphs only for
  how to size it or when to change it (what to count, what happens when it is too small).
- Do not justify the grouping of options or restate the plugin description; that belongs in the
  plugin header (commit c35152b6f5f, "mpipe: base: Shorten the MPIPE_BASE help text").
- Keep `zephyr-keep-sorted` blocks sorted (`base/CMakeLists.txt`).

## Build and test

All tests run on `native_sim/native/64`. From the workspace's `zephyr/` directory:

    west twister -p native_sim/native/64 -T tests/subsys/mpipe
    west twister -p native_sim/native/64 -s tests/subsys/mpipe/unit/mpipe.unit

- `unit`: value, structure, pad, element, bin, buffer, thread, transform.
- `pipeline`: end-to-end runs with `mpipe_fake_src` (`CONFIG_MPIPE_FAKE_SRC`).
- `base`: queue and app_src/app_sink.
- `utils`: dump and player.
- `build_all`: build-only scenarios, one per plugin. Extend it when adding a plugin or a Kconfig
  option that gates code.

Samples need real hardware (cameras, displays, DMIC/I2S codecs, SD cards); most board
overlays target NXP boards such as `mimxrt700_evk`. Build them with `west build -b <board>
samples/subsys/mpipe/<sample>` and state when a change was not run on hardware.

To debug a graph, enable `CONFIG_MPIPE_DUMP` (and `CONFIG_MPIPE_PLAYER_DUMP_ON_STATE_CHANGE`
with the player), then extract and render the DOT output on the host:

    sed -n '/^digraph/,/^}/p' log > pipe.dot && dot -Tpng pipe.dot -o pipe.png
