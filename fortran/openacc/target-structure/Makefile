FC       := nvfortran
TARGET   := test.x

OPTFLAGS := -O3
OACCFLAGS := -acc=gpu
INFO     := -Minfo=accel
FFLAGS   := $(OPTFLAGS) $(OACCFLAGS) $(INFO)

SRCS     := foo_mod.f90 main.f90
OBJS     := foo_mod.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@

main.o: foo_mod.o

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *.mod

rebuild: clean all

.PHONY: all clean
