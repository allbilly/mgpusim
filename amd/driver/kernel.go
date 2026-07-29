package driver

import (
	"encoding/binary"
	"log"
	"reflect"

	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/driver/internal"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/kernels"
)

// EnqueueLaunchKernel schedules kernel to be launched later
func (d *Driver) EnqueueLaunchKernel(
	queue *CommandQueue,
	co *insts.KernelCodeObject,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) {
	dev := d.devices[queue.GPUID]

	if dev.Type == internal.DeviceTypeUnifiedGPU {
		d.enqueueLaunchUnifiedKernel(queue, co, gridSize, wgSize, kernelArgs)
	} else {
		dCoData, cached := d.codeObjGPUAddrs[co]
		if !cached {
			dCoData = d.AllocateMemory(queue.Context, uint64(len(co.Data)))
			d.codeObjGPUAddrs[co] = dCoData
		}

		dKernArgData := d.AllocateMemory(queue.Context, co.KernargSegmentByteSize)

		packet := kernels.HsaKernelDispatchPacket{}
		dPacket := d.AllocateMemory(queue.Context, uint64(binary.Size(packet)))

		aqlPacket := d.createAQLPacket(gridSize, wgSize, dCoData, dKernArgData)
		newKernelArgs := d.prepareLocalMemory(co, kernelArgs, aqlPacket)
		d.prepareScratch(queue.Context, co, gridSize, aqlPacket)

		if !cached {
			d.EnqueueMemCopyH2D(queue, dCoData, co.Data)
		}
		d.EnqueueMemCopyH2D(queue, dKernArgData, d.kernargBytes(co, newKernelArgs))
		d.EnqueueMemCopyH2D(queue, dPacket, aqlPacket)

		d.enqueueLaunchKernelCommand(queue, co, aqlPacket, dPacket)
	}
}

// prepareScratch allocates the private/scratch segment when the kernel needs
// one (VGPR spill via MUBUF, etc.) and records its VA on the AQL packet.
func (d *Driver) prepareScratch(
	ctx *Context,
	co *insts.KernelCodeObject,
	gridSize [3]uint32,
	packet *kernels.HsaKernelDispatchPacket,
) {
	if co.PrivateSegmentByteSize == 0 {
		return
	}
	nWI := uint64(gridSize[0]) * uint64(gridSize[1]) * uint64(gridSize[2])
	size := nWI * uint64(co.PrivateSegmentByteSize)
	scratch := d.AllocateMemory(ctx, size)
	packet.PrivateSegmentSize = co.PrivateSegmentByteSize
	packet.ScratchAddress = uint64(scratch)
}

func (d *Driver) allocateGPUMemory(
	ctx *Context,
	co *insts.KernelCodeObject,
) (dCoData, dKernArgData, dPacket Ptr) {
	dCoData = d.AllocateMemory(ctx, uint64(len(co.Data)))
	dKernArgData = d.AllocateMemory(ctx, co.KernargSegmentByteSize)

	packet := kernels.HsaKernelDispatchPacket{}
	dPacket = d.AllocateMemory(ctx, uint64(binary.Size(packet)))

	return dCoData, dKernArgData, dPacket
}

func (d *Driver) prepareLocalMemory(
	co *insts.KernelCodeObject,
	kernelArgs interface{},
	packet *kernels.HsaKernelDispatchPacket,
) (newKernelArgs interface{}) {
	argsType := reflect.TypeOf(kernelArgs)
	argsValue := reflect.ValueOf(kernelArgs)

	// Handle pointer/slice types
	if argsType.Kind() == reflect.Ptr {
		// Create a new instance of the pointed-to type
		newKernelArgs = reflect.New(argsType.Elem()).Interface()
		reflect.ValueOf(newKernelArgs).Elem().Set(argsValue.Elem())
	} else if argsType.Kind() == reflect.Slice {
		// For slices, just pass through
		newKernelArgs = kernelArgs
	} else {
		// For structs, create a new instance
		newKernelArgs = reflect.New(argsType).Interface()
		reflect.ValueOf(newKernelArgs).Elem().Set(argsValue)
	}

	ldsSize := co.GroupSegmentByteSize

	if reflect.TypeOf(newKernelArgs).Kind() == reflect.Slice {
		// From server, do nothing
	} else {
		kernArgStruct := reflect.ValueOf(newKernelArgs).Elem()
		for i := 0; i < kernArgStruct.NumField(); i++ {
			// Skip unexported fields (padding fields)
			if !kernArgStruct.Field(i).CanInterface() {
				continue
			}
			arg := kernArgStruct.Field(i).Interface()

			switch ldsPtr := arg.(type) {
			case LocalPtr:
				kernArgStruct.Field(i).SetUint(uint64(ldsSize))
				ldsSize += uint32(ldsPtr)
			}
		}
	}

	packet.GroupSegmentSize = ldsSize

	return newKernelArgs
}

func (d *Driver) kernargBytes(
	co *insts.KernelCodeObject,
	kernelArgs interface{},
) interface{} {
	if co.Version == insts.CodeObjectV5 || co.KernargSegmentByteSize > 0 {
		return d.prepareKernargBytes(co, kernelArgs)
	}
	return kernelArgs
}

// prepareKernargBytes serializes kernel arguments to a byte slice.
// Fields are placed using Go struct layout offsets so alignment padding is
// preserved (e.g., V5 code objects with 88-byte kernarg segments).
func (d *Driver) prepareKernargBytes(
	co *insts.KernelCodeObject,
	kernelArgs interface{},
) []byte {
	argsValue := reflect.ValueOf(kernelArgs)
	if argsValue.Kind() == reflect.Ptr {
		argsValue = argsValue.Elem()
	}
	argsType := argsValue.Type()

	kernargSize := int(co.KernargSegmentByteSize)
	if kernargSize == 0 {
		kernargSize = int(argsType.Size())
	}
	buf := make([]byte, kernargSize)

	for i := 0; i < argsType.NumField(); i++ {
		field := argsType.Field(i)
		if !argsValue.Field(i).CanInterface() {
			continue
		}

		fieldOffset := int(field.Offset)
		fieldSize := int(field.Type.Size())
		if fieldOffset+fieldSize > kernargSize {
			break
		}

		marshaled := make([]byte, fieldSize)
		switch v := argsValue.Field(i).Interface().(type) {
		case Ptr:
			binary.LittleEndian.PutUint64(marshaled, uint64(v))
		case uint64:
			binary.LittleEndian.PutUint64(marshaled, v)
		case int64:
			binary.LittleEndian.PutUint64(marshaled, uint64(v))
		case uint32:
			binary.LittleEndian.PutUint32(marshaled, v)
		case int32:
			binary.LittleEndian.PutUint32(marshaled, uint32(v))
		case uint16:
			binary.LittleEndian.PutUint16(marshaled[0:2], v)
		case int16:
			binary.LittleEndian.PutUint16(marshaled[0:2], uint16(v))
		case uint8:
			marshaled[0] = v
		case int8:
			marshaled[0] = byte(v)
		case LocalPtr:
			binary.LittleEndian.PutUint32(marshaled, uint32(v))
		default:
			data := argsValue.Field(i).Bytes()
			copy(marshaled, data)
		}

		copy(buf[fieldOffset:fieldOffset+fieldSize], marshaled)
	}

	return buf
}

// LaunchKernel is an easy way to run a kernel on the GCN3 simulator. It
// launches the kernel immediately.
func (d *Driver) LaunchKernel(
	ctx *Context,
	co *insts.KernelCodeObject,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) {
	queue := d.CreateCommandQueue(ctx)
	d.EnqueueLaunchKernel(queue, co, gridSize, wgSize, kernelArgs)
	d.DrainCommandQueue(queue)
}

// EnqueueLaunchKernelWithKernarg is like EnqueueLaunchKernel but allows
// specifying a pre-allocated kernarg segment pointer. This is useful for
// V5 code objects that have specific kernarg size requirements.
func (d *Driver) EnqueueLaunchKernelWithKernarg(
	queue *CommandQueue,
	co *insts.KernelCodeObject,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
	kernargPtr Ptr,
) {
	dev := d.devices[queue.GPUID]

	if dev.Type == internal.DeviceTypeUnifiedGPU {
		log.Panic("Unified GPU not supported for this function")
	}

	dCoData, cached := d.codeObjGPUAddrs[co]
	if !cached {
		dCoData = d.AllocateMemory(queue.Context, uint64(len(co.Data)))
		d.codeObjGPUAddrs[co] = dCoData
	}

	// Use provided kernarg pointer if specified
	if kernargPtr == 0 {
		kernargPtr = d.AllocateMemory(queue.Context, co.KernargSegmentByteSize)
	}

	packet := kernels.HsaKernelDispatchPacket{}
	dPacket := d.AllocateMemory(queue.Context, uint64(binary.Size(packet)))

	aqlPacket := d.createAQLPacket(gridSize, wgSize, dCoData, kernargPtr)
	
	// Only prepare local memory if kernelArgs is provided
	if kernelArgs != nil {
		_ = d.prepareLocalMemory(co, kernelArgs, aqlPacket)
	} else {
		aqlPacket.GroupSegmentSize = co.GroupSegmentByteSize
	}

	if !cached {
		d.EnqueueMemCopyH2D(queue, dCoData, co.Data)
	}
	// Note: kernarg data should already be copied before calling this function
	d.EnqueueMemCopyH2D(queue, dPacket, aqlPacket)

	d.enqueueLaunchKernelCommand(queue, co, aqlPacket, dPacket)
}

func (d *Driver) createAQLPacket(
	gridSize [3]uint32,
	wgSize [3]uint16,
	dCoData Ptr,
	dKernArgData Ptr,
) *kernels.HsaKernelDispatchPacket {
	packet := new(kernels.HsaKernelDispatchPacket)
	packet.GridSizeX = gridSize[0]
	packet.GridSizeY = gridSize[1]
	packet.GridSizeZ = gridSize[2]
	packet.WorkgroupSizeX = wgSize[0]
	packet.WorkgroupSizeY = wgSize[1]
	packet.WorkgroupSizeZ = wgSize[2]
	packet.KernelObject = uint64(dCoData)
	packet.KernargAddress = uint64(dKernArgData)
	return packet
}

func (d *Driver) enqueueLaunchKernelCommand(
	queue *CommandQueue,
	co *insts.KernelCodeObject,
	packet *kernels.HsaKernelDispatchPacket,
	dPacket Ptr,
) {
	cmd := &LaunchKernelCommand{
		ID:         timing.GetIDGenerator().Generate(),
		CodeObject: co,
		DPacket:    dPacket,
		Packet:     packet,
	}
	d.Enqueue(queue, cmd)
}

func (d *Driver) enqueueLaunchUnifiedKernelCommand(
	queue *CommandQueue,
	co *insts.KernelCodeObject,
	packet []*kernels.HsaKernelDispatchPacket,
	dPacket []Ptr,
) {
	cmd := &LaunchUnifiedMultiGPUKernelCommand{
		ID:           timing.GetIDGenerator().Generate(),
		CodeObject:   co,
		DPacketArray: dPacket,
		PacketArray:  packet,
	}
	d.Enqueue(queue, cmd)
}
func (d *Driver) enqueueLaunchUnifiedKernel(
	queue *CommandQueue,
	co *insts.KernelCodeObject,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) {
	dev := d.devices[queue.GPUID]
	initGPUID := queue.Context.currentGPUID
	queueArray := make([]*CommandQueue, len(dev.UnifiedGPUIDs)+1)
	dCoDataArray := make([]Ptr, len(dev.UnifiedGPUIDs)+1)
	dKernArgDataArray := make([]Ptr, len(dev.UnifiedGPUIDs)+1)
	dPacketArray := make([]Ptr, len(dev.UnifiedGPUIDs)+1)
	packetArray := make([]*kernels.HsaKernelDispatchPacket, len(dev.UnifiedGPUIDs)+1)
	// fmt.Printf("# of GPUs : %v \n", len(dev.UnifiedGPUIDs))

	for i, gpuID := range dev.UnifiedGPUIDs {
		queueArray[i] = queue
		queueArray[i].Context.currentGPUID = gpuID
		dCoData, dKernArgData, dPacket := d.allocateGPUMemory(queue.Context, co)

		packet := d.createAQLPacket(gridSize, wgSize, dCoData, dKernArgData)
		newKernelArgs := d.prepareLocalMemory(co, kernelArgs, packet)
		d.prepareScratch(queue.Context, co, gridSize, packet)

		d.EnqueueMemCopyH2D(queue, dCoData, co.Data)
		d.EnqueueMemCopyH2D(queue, dKernArgData, d.kernargBytes(co, newKernelArgs))
		d.EnqueueMemCopyH2D(queue, dPacket, packet)

		dCoDataArray[i] = dCoData
		dKernArgDataArray[i] = dKernArgData
		dPacketArray[i] = dPacket
		packetArray[i] = packet
		// fmt.Printf("packetArray: %v \n", packetArray[i])
	}

	queue.Context.currentGPUID = initGPUID
	d.enqueueLaunchUnifiedKernelCommand(queue, co, packetArray, dPacketArray)
}
