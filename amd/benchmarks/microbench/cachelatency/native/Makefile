# Container-based HIP compilation for the cache_latency microbenchmark.
# hipcc emits a relocatable device object (.o) and a linked loadable code
# object (.out); the latter is the HSACO accepted by both HIP and MGPUSim.

DOCKER_IMAGE = rocm/dev-ubuntu-24.04:7.2.4
GPU_TARGET = gfx942
SRC = cache_latency.cpp
GPU_OBJ = cache_latency-hip-amdgcn-amd-amdhsa-$(GPU_TARGET).out
HOST_OBJ = cache_latency_host.o
HSACO = cache_latency.hsaco
DISASM = cache_latency.disasm
CONTAINER_RUNTIME ?= docker

# --platform linux/amd64 required for Apple Silicon Macs
DOCKER_RUN = $(CONTAINER_RUNTIME) run --rm --platform linux/amd64 -v $(PWD):/work -w /work $(DOCKER_IMAGE)

.PHONY: all clean clean-all regen-disasm

all: $(HSACO)

# Compile HIP source with --save-temps to get intermediate GPU object file
$(HSACO): $(SRC)
	$(DOCKER_RUN) hipcc -O3 --save-temps -c --offload-arch=$(GPU_TARGET) $< -o $(HOST_OBJ)
	cp $(GPU_OBJ) $(HSACO)

# Regenerate LLVM disassembly (requires Docker)
regen-disasm: $(HSACO)
	$(DOCKER_RUN) /opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$(GPU_TARGET) $(HSACO) > $(DISASM)
	@echo "Updated $(DISASM)"

clean:
	rm -f *.ll *.bc *.s *.o *.cui *.fatbin *.hipfb
	@echo "Note: $(HSACO) and $(DISASM) are preserved (checked into git)"

clean-all: clean
	rm -f $(HSACO) $(DISASM)
