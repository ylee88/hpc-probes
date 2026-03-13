CUDA_HOME ?= /opt/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

TARGET    := dgemm-lt-fixedpoint-benchmark.x
SRC       := dgemm-lt-fixedpoint-benchmark.cu

NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_86
LDFLAGS   ?= -lcublas -lcublasLt

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
