package kernels

import "encoding/binary"

// GFX9 SQ_BUF_RSRC word3 used for private/scratch: XYZW sel, float NF,
// DATA_FORMAT_32, ADD_TID_ENABLE (bit 23). Matches amdkfd CWSR trap handler.
const privateSegmentRSRCWord3 = 0x00807FAC

// WritePrivateSegmentBufferDesc fills a 16-byte V# buffer descriptor for the
// private/scratch segment. Stride is per-work-item bytes; ADD_TID_ENABLE makes
// MUBUF off-mode addresses add lane_id*stride.
func WritePrivateSegmentBufferDesc(dst []byte, scratchVA uint64, stride uint32) {
	if len(dst) < 16 {
		panic("private segment buffer dst too small")
	}
	binary.LittleEndian.PutUint32(dst[0:4], uint32(scratchVA))
	word1 := uint32(scratchVA>>32)&0xffff | (stride << 16)
	binary.LittleEndian.PutUint32(dst[4:8], word1)
	binary.LittleEndian.PutUint32(dst[8:12], 0xffffffff) // NUM_RECORDS (bytes)
	binary.LittleEndian.PutUint32(dst[12:16], privateSegmentRSRCWord3)
}

// WorkGroupFlatID is the launch-order index of a work-group in the grid.
func WorkGroupFlatID(wg *WorkGroup) int {
	numWGX := (int(wg.Packet.GridSizeX) + wg.SizeX - 1) / wg.SizeX
	numWGY := (int(wg.Packet.GridSizeY) + wg.SizeY - 1) / wg.SizeY
	return wg.IDX + wg.IDY*numWGX + wg.IDZ*numWGX*numWGY
}

// PrivateSegmentWaveByteOffset is the SPI wave scratch offset for ADD_TID
// addressing: (wg_flat * wg_size + first_local_flat) * stride. Lane tid then
// adds tid*stride so concurrent WGs do not share scratch slots.
func PrivateSegmentWaveByteOffset(
	wg *WorkGroup, firstLocalFlatID int, stride uint32,
) uint32 {
	wgSize := wg.SizeX * wg.SizeY * wg.SizeZ
	return uint32(WorkGroupFlatID(wg)*wgSize+firstLocalFlatID) * stride
}
