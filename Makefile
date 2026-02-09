CC      := gcc
CFLAGS  := -Iinclude -O2 -Wall -Wextra -std=c11
TARGET  := main
SRCS    := src/main.c src/util.c src/matrix_multiply_io.c src/formats.c src/spmv.c src/bench.c
OBJS    := $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
