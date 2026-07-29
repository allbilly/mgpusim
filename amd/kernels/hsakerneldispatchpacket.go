package kernels

// An HsaKernelDispatchPacket is an AQL packet for launching a kernel on a
// GPU.
type HsaKernelDispatchPacket struct {
	Header             uint16
	Setup              uint16
	WorkgroupSizeX     uint16
	WorkgroupSizeY     uint16
	WorkgroupSizeZ     uint16
	reserverd0         uint16
	GridSizeX          uint32
	GridSizeY          uint32
	GridSizeZ          uint32
	PrivateSegmentSize uint32
	GroupSegmentSize   uint32
	KernelObject       uint64
	KernargAddress     uint64
	// ScratchAddress is the GPU VA of the private/scratch segment for this
	// dispatch (AQL reserved2). Zero when the kernel needs no scratch.
	ScratchAddress   uint64
	CompletionSignal uint64
}
