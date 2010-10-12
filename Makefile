# /----------------------------------------------------------
# | General setup
# +----------------------------------------------------------
# |
CC = cc
CFLAGS = -std=c99 -g -Wall -Werror -D_XOPEN_SOURCE=500
LDFLAGS = -lOpenCL -lm
# |
# \----------------------------------------------------------

# /----------------------------------------------------------
# | Compilation rules
# +----------------------------------------------------------
# |

.PHONY:	all
all:	hello-gpu gpu-wave

# Compile a C version (using basic_dgemm.c, in this case):

hello-gpu: hello-gpu.o cl-helper.o
	$(CC) -o $@ $^ $(LDFLAGS)

gpu-wave: gpu-wave.o cl-helper.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Generic Rules
%.o:%.c
	$(CC) -c $(CFLAGS) $(OPTFLAGS) $<
# |
# \----------------------------------------------------------

# /----------------------------------------------------------
# | Clean-up rules
# +----------------------------------------------------------

.PHONY:	clean
clean:
	rm -f hello-gpu gpu-wave *.o
# |
# \----------------------------------------------------------

