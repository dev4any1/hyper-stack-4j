/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.hyperstack4j.node;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Reads GGUF v2/v3 files and exposes tensor data as float[] arrays.
 *
 * Supported quantisation types: F32 — lossless passthrough F16 — half-precision
 * → float32 (IEEE 754 bit manipulation, no JNI) BF16 — bfloat16 → float32 Q8_0
 * — 8-bit symmetric, block size 32 Q4_0 — 4-bit symmetric, block size 32 Q4_K —
 * 4-bit with per-superblock scale/min, block size 256 (Q4_K_M uses this) Q6_K —
 * 6-bit with per-superblock scale, block size 256
 *
 * Thread-safe after construction — tensor data is loaded on demand then cached.
 *
 * Usage: GgufReader r =
 * GgufReader.open(Path.of("/models/TinyLlama.Q4_K_M.gguf")); LlamaConfig cfg =
 * LlamaConfig.from(r); float[] w = r.tensor("blk.0.attn_q.weight");
 */
public final class GgufReader implements AutoCloseable {

	private static final Logger log = Logger.getLogger(GgufReader.class.getName());

	private static final int GGUF_MAGIC = 0x46554747; // "GGUF"
	private static final int ALIGNMENT = 32;

	// ── GGML quantisation type IDs ───────────────────────────────────────────
	private static final int GGML_TYPE_F32 = 0;
	private static final int GGML_TYPE_F16 = 1;
	private static final int GGML_TYPE_Q4_0 = 2;
	private static final int GGML_TYPE_Q8_0 = 8;
	private static final int GGML_TYPE_Q4_K = 12;
	private static final int GGML_TYPE_Q6_K = 14;
	private static final int GGML_TYPE_BF16 = 30;

	// Metadata value type IDs
	private static final int GGUF_METADATA_VALUE_TYPE_UINT8 = 0;
	private static final int GGUF_METADATA_VALUE_TYPE_INT8 = 1;
	private static final int GGUF_METADATA_VALUE_TYPE_UINT16 = 2;
	private static final int GGUF_METADATA_VALUE_TYPE_INT16 = 3;
	private static final int GGUF_METADATA_VALUE_TYPE_UINT32 = 4;
	private static final int GGUF_METADATA_VALUE_TYPE_INT32 = 5;
	private static final int GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6;
	private static final int GGUF_METADATA_VALUE_TYPE_BOOL = 7;
	private static final int GGUF_METADATA_VALUE_TYPE_STRING = 8;
	private static final int GGUF_METADATA_VALUE_TYPE_ARRAY = 9;
	private static final int GGUF_METADATA_VALUE_TYPE_UINT64 = 10;
	private static final int GGUF_METADATA_VALUE_TYPE_INT64 = 11;
	private static final int GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12;

	private final FileChannel channel;
	private final Map<String, Object> metadata = new HashMap<>();
	private final Map<String, TensorInfo> tensors = new HashMap<>();
	private final long dataOffset; // byte offset where tensor data starts
	private final Map<String, float[]> cache = new HashMap<>();

	// ── Constructor / factory ─────────────────────────────────────────────────

	private GgufReader(FileChannel channel, Map<String, Object> metadata, Map<String, TensorInfo> tensors,
			long dataOffset) {
		this.channel = channel;
		this.metadata.putAll(metadata);
		this.tensors.putAll(tensors);
		this.dataOffset = dataOffset;
	}

	public static GgufReader open(Path file) throws IOException {
		FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
		ByteBuffer header = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
		channel.read(header, 0);
		header.flip();

		int magic = header.getInt();
		int version = header.getInt();
		long tensorCount = header.getLong();
		long kvCount = header.getLong();

		if (magic != GGUF_MAGIC)
			throw new IOException("Not a GGUF file (magic=0x" + Integer.toHexString(magic) + ")");
		if (version < 2 || version > 3)
			throw new IOException("Unsupported GGUF version: " + version);

		log.info("GGUF v" + version + " — tensors=" + tensorCount + " metadata=" + kvCount);

		// Parse using a streaming position tracker
		long[] pos = { 24L };

		// Read metadata
		Map<String, Object> metadata = new HashMap<>();
		for (long i = 0; i < kvCount; i++) {
			String key = readString(channel, pos);
			int vtype = readInt32(channel, pos);
			Object value = readMetadataValue(channel, pos, vtype);
			metadata.put(key, value);
		}

		// Read tensor info
		Map<String, TensorInfo> tensors = new HashMap<>();
		for (long i = 0; i < tensorCount; i++) {
			String name = readString(channel, pos);
			int ndims = readInt32(channel, pos);
			long[] dims = new long[ndims];
			for (int d = 0; d < ndims; d++)
				dims[d] = readUInt64(channel, pos);
			int type = readInt32(channel, pos);
			long offset = readUInt64(channel, pos);
			long nelems = 1;
			for (long d : dims)
				nelems *= d;
			tensors.put(name, new TensorInfo(name, dims, type, offset, nelems));
		}

		// Align to ALIGNMENT bytes
		long aligned = ((pos[0] + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

		log.info("Data section starts at byte " + aligned);
		return new GgufReader(channel, metadata, tensors, aligned);
	}

	// ── Public API ────────────────────────────────────────────────────────────

	/** Get a metadata value (String, Number, Boolean, Object[], …). */
	public Object meta(String key) {
		return metadata.get(key);
	}

	public String metaString(String key) {
		Object v = metadata.get(key);
		return v instanceof String s ? s : null;
	}

	public long metaLong(String key, long def) {
		Object v = metadata.get(key);
		return v instanceof Number n ? n.longValue() : def;
	}

	public int metaInt(String key, int def) {
		Object v = metadata.get(key);
		return v instanceof Number n ? n.intValue() : def;
	}

	public float metaFloat(String key, float def) {
		Object v = metadata.get(key);
		return v instanceof Number n ? n.floatValue() : def;
	}

	public boolean hasTensor(String name) {
		return tensors.containsKey(name);
	}

	public Map<String, Object> allMetadata() {
		return java.util.Collections.unmodifiableMap(metadata);
	}

	/**
	 * Load and dequantize a tensor to float[]. Results are cached — subsequent
	 * calls with the same name are free.
	 */
	public float[] tensor(String name) throws IOException {
		float[] cached = cache.get(name);
		if (cached != null)
			return cached;

		TensorInfo info = tensors.get(name);
		if (info == null)
			throw new IllegalArgumentException(
					"Tensor not found: " + name + "  (available: " + tensors.size() + " tensors)");

		float[] data = loadTensor(info);
		cache.put(name, data);
		return data;
	}

	@Override
	public void close() throws IOException {
		channel.close();
	}

	// ── Tensor loading + dequantisation ───────────────────────────────────────

	private float[] loadTensor(TensorInfo info) throws IOException {
		return switch (info.type) {
		case GGML_TYPE_F32 -> loadF32(info);
		case GGML_TYPE_F16 -> loadF16(info);
		case GGML_TYPE_BF16 -> loadBF16(info);
		case GGML_TYPE_Q8_0 -> loadQ8_0(info);
		case GGML_TYPE_Q4_0 -> loadQ4_0(info);
		case GGML_TYPE_Q4_K -> loadQ4_K(info);
		case GGML_TYPE_Q6_K -> loadQ6_K(info);
		default -> throw new UnsupportedOperationException(
				"Unsupported tensor type " + info.type + " for tensor " + info.name);
		};
	}

	private float[] loadF32(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		ByteBuffer buf = readBytes(info.offset, (long) n * 4);
		float[] out = new float[n];
		buf.asFloatBuffer().get(out);
		return out;
	}

	private float[] loadF16(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		ByteBuffer buf = readBytes(info.offset, (long) n * 2);
		float[] out = new float[n];
		for (int i = 0; i < n; i++)
			out[i] = f16ToF32(buf.getShort());
		return out;
	}

	private float[] loadBF16(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		ByteBuffer buf = readBytes(info.offset, (long) n * 2);
		float[] out = new float[n];
		for (int i = 0; i < n; i++) {
			int bits = (buf.getShort() & 0xFFFF) << 16;
			out[i] = Float.intBitsToFloat(bits);
		}
		return out;
	}

	// Q8_0: blocks of 32 elements, 2-byte f16 scale + 32 signed bytes
	private float[] loadQ8_0(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int blockSize = 32;
		int blockBytes = 2 + blockSize; // scale f16 + 32 x int8
		int nBlocks = n / blockSize;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			float scale = f16ToF32(buf.getShort());
			for (int i = 0; i < blockSize; i++)
				out[oi++] = scale * buf.get(); // signed byte
		}
		return out;
	}

	// Q4_0: blocks of 32 elements, 2-byte f16 scale + 16 packed nibbles
	private float[] loadQ4_0(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int blockSize = 32;
		int blockBytes = 2 + blockSize / 2;
		int nBlocks = n / blockSize;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			float scale = f16ToF32(buf.getShort());
			// Low nibble = first 16, high nibble = second 16
			byte[] qs = new byte[16];
			buf.get(qs);
			for (int i = 0; i < 16; i++)
				out[oi++] = scale * ((qs[i] & 0xF) - 8);
			for (int i = 0; i < 16; i++)
				out[oi++] = scale * ((qs[i] >> 4 & 0xF) - 8);
		}
		return out;
	}

	// Q4_K: superblocks of 256 elements
	// [d:f16][dmin:f16][scales:12 bytes][qs:128 bytes] = 144 bytes per 256 elements
	private float[] loadQ4_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 144;
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] qs = new byte[128];
		byte[] sc = new byte[12];

		for (int b = 0; b < nBlocks; b++) {
			float d = f16ToF32(buf.getShort());
			float dmin = f16ToF32(buf.getShort());
			buf.get(sc);
			buf.get(qs);

			// 8 sub-blocks of 32 elements each, grouped as 4 pairs of 64
			int qi = 0;
			for (int g = 0; g < QK_K; g += 64) {
				// pair index within the 8 sub-blocks (g/32 and g/32+1)
				int s0 = g / 32;
				int s1 = s0 + 1;
				float scale0 = d * getScale4K(sc, s0);
				float min0 = dmin * getMin4K(sc, s0);
				float scale1 = d * getScale4K(sc, s1);
				float min1 = dmin * getMin4K(sc, s1);

				// First 32: low nibbles of qs[qi..qi+32)
				for (int i = 0; i < 32; i++)
					out[oi++] = scale0 * (qs[qi + i] & 0x0F) - min0;
				// Second 32: high nibbles of qs[qi..qi+32)
				for (int i = 0; i < 32; i++)
					out[oi++] = scale1 * ((qs[qi + i] >> 4) & 0x0F) - min1;
				qi += 32;
			}
		}
		return out;
	}

	/**
	 * Extract 6-bit scale[j] from Q4_K scales block (12 bytes, 8 scales + 8 mins
	 * each 6-bit).
	 */
	private static int getScale4K(byte[] sc, int j) {
		if (j < 4)
			return sc[j] & 0x3F;
		return ((sc[j + 4] & 0x0F) | ((sc[j - 4] & 0xC0) >> 2)) & 0x3F;
	}

	/** Extract 6-bit min[j] from Q4_K scales block. */
	private static int getMin4K(byte[] sc, int j) {
		if (j < 4)
			return sc[j + 4] & 0x3F;
		// sc bytes are signed in Java — mask with 0xFF before >> 4 to prevent
		// arithmetic sign-extension corrupting the upper bits of the result.
		return (((sc[j + 4] & 0xFF) >> 4) | ((sc[j] & 0xC0) >> 2)) & 0x3F;
	}

	// Q6_K: superblocks of 256 elements
	// [ql:128 bytes][qh:64 bytes][scales:16 bytes][d:f16] = 210 bytes per 256
	// elements
	private float[] loadQ6_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 210;
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] ql = new byte[128];
		byte[] qh = new byte[64];
		byte[] sc = new byte[16];

		for (int b = 0; b < nBlocks; b++) {
			buf.get(ql);
			buf.get(qh);
			buf.get(sc);
			float d = f16ToF32(buf.getShort());

			// Port of llama.cpp dequantize_row_q6_K.
			// Each 256-element block is split into two halves of 128 elements.
			// Within each half, l iterates 0..31 and produces four outputs:
			// out[l+ 0] ← ql[qlBase+l] low nibble | qh[qhBase+l] bits 1:0 → sub-block
			// sc[scBase + l/16]
			// out[l+ 32] ← ql[qlBase+l+32] low nibble | qh[qhBase+l] bits 3:2 → sc[scBase +
			// l/16 + 2]
			// out[l+ 64] ← ql[qlBase+l] high nibble | qh[qhBase+l] bits 5:4 → sc[scBase +
			// l/16 + 4]
			// out[l+ 96] ← ql[qlBase+l+32] high nibble | qh[qhBase+l] bits 7:6 → sc[scBase
			// + l/16 + 6]
			// All four share the SAME qh byte qh[qhBase+l]; the earlier flat loop
			// used hi=i/4 which is wrong for outputs at l+32, l+64, l+96.
			for (int half = 0; half < 2; half++) {
				int qlBase = half * 64;
				int qhBase = half * 32;
				int scBase = half * 8;
				for (int l = 0; l < 32; l++) {
					int is = l / 16;
					int qlL = ql[qlBase + l] & 0xFF;
					int qlL2 = ql[qlBase + l + 32] & 0xFF;
					int qhL = qh[qhBase + l] & 0xFF;

					int q1 = (qlL & 0x0F) | (((qhL >> 0) & 3) << 4);
					q1 -= 32;
					int q2 = (qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4);
					q2 -= 32;
					int q3 = (qlL >> 4) | (((qhL >> 4) & 3) << 4);
					q3 -= 32;
					int q4 = (qlL2 >> 4) | (((qhL >> 6) & 3) << 4);
					q4 -= 32;

					// sc[] is int8 — Java bytes are signed, which is what we want.
					float d1 = d * sc[scBase + is];
					float d2 = d * sc[scBase + is + 2];
					float d3 = d * sc[scBase + is + 4];
					float d4 = d * sc[scBase + is + 6];

					out[oi + l] = d1 * q1;
					out[oi + l + 32] = d2 * q2;
					out[oi + l + 64] = d3 * q3;
					out[oi + l + 96] = d4 * q4;
				}
				oi += 128;
			}
		}
		return out;
	}

	// ── I/O helpers ───────────────────────────────────────────────────────────

	private ByteBuffer readBytes(long tensorOffset, long byteCount) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate((int) byteCount).order(ByteOrder.LITTLE_ENDIAN);
		int read = channel.read(buf, dataOffset + tensorOffset);
		if (read != byteCount)
			throw new IOException("Expected " + byteCount + " bytes, got " + read);
		buf.flip();
		return buf;
	}

	private static String readString(FileChannel ch, long[] pos) throws IOException {
		long len = readUInt64(ch, pos);
		ByteBuffer buf = ByteBuffer.allocate((int) len);
		ch.read(buf, pos[0]);
		pos[0] += len;
		return new String(buf.array(), java.nio.charset.StandardCharsets.UTF_8);
	}

	private static int readInt32(FileChannel ch, long[] pos) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
		ch.read(buf, pos[0]);
		pos[0] += 4;
		buf.flip();
		return buf.getInt();
	}

	private static long readUInt64(FileChannel ch, long[] pos) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
		ch.read(buf, pos[0]);
		pos[0] += 8;
		buf.flip();
		return buf.getLong();
	}

	private static Object readMetadataValue(FileChannel ch, long[] pos, int vtype) throws IOException {
		return switch (vtype) {
		case GGUF_METADATA_VALUE_TYPE_UINT8, GGUF_METADATA_VALUE_TYPE_INT8 -> {
			ByteBuffer b = ByteBuffer.allocate(1);
			ch.read(b, pos[0]);
			pos[0]++;
			b.flip();
			yield (int) b.get();
		}
		case GGUF_METADATA_VALUE_TYPE_UINT16, GGUF_METADATA_VALUE_TYPE_INT16 -> {
			ByteBuffer b = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(b, pos[0]);
			pos[0] += 2;
			b.flip();
			yield (int) b.getShort();
		}
		case GGUF_METADATA_VALUE_TYPE_UINT32, GGUF_METADATA_VALUE_TYPE_INT32 -> {
			int v = readInt32(ch, pos);
			yield v;
		}
		case GGUF_METADATA_VALUE_TYPE_FLOAT32 -> {
			ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(b, pos[0]);
			pos[0] += 4;
			b.flip();
			yield b.getFloat();
		}
		case GGUF_METADATA_VALUE_TYPE_BOOL -> {
			ByteBuffer b = ByteBuffer.allocate(1);
			ch.read(b, pos[0]);
			pos[0]++;
			b.flip();
			yield b.get() != 0;
		}
		case GGUF_METADATA_VALUE_TYPE_STRING -> readString(ch, pos);
		case GGUF_METADATA_VALUE_TYPE_UINT64, GGUF_METADATA_VALUE_TYPE_INT64 -> readUInt64(ch, pos);
		case GGUF_METADATA_VALUE_TYPE_FLOAT64 -> {
			ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(b, pos[0]);
			pos[0] += 8;
			b.flip();
			yield b.getDouble();
		}
		case GGUF_METADATA_VALUE_TYPE_ARRAY -> readArray(ch, pos);
		default -> throw new IOException("Unknown metadata type: " + vtype);
		};
	}

	private static Object[] readArray(FileChannel ch, long[] pos) throws IOException {
		int elemType = readInt32(ch, pos);
		long count = readUInt64(ch, pos);
		Object[] arr = new Object[(int) count];
		for (int i = 0; i < count; i++)
			arr[i] = readMetadataValue(ch, pos, elemType);
		return arr;
	}

	// ── F16 → F32 (pure Java, no JNI) ────────────────────────────────────────

	static float f16ToF32(short bits) {
		int s = (bits >> 15) & 1;
		int e = (bits >> 10) & 0x1F;
		int m = bits & 0x3FF;
		int fBits;
		if (e == 0) {
			if (m == 0) {
				fBits = s << 31;
			} else { // subnormal → normalise
				int exp = -14;
				while ((m & 0x400) == 0) {
					m <<= 1;
					exp--;
				}
				m &= 0x3FF;
				fBits = (s << 31) | ((exp + 127) << 23) | (m << 13);
			}
		} else if (e == 31) {
			fBits = (s << 31) | 0x7F800000 | (m << 13); // ±inf or NaN
		} else {
			fBits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
		}
		return Float.intBitsToFloat(fBits);
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	record TensorInfo(String name, long[] dims, int type, long offset, long nelems) {
	}
}