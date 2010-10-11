# Makefile for the optimize matrix multiply assignment

# /----------------------------------------------------------
# | General setup
# +----------------------------------------------------------
# |
CC = cc
CFLAGS = -std=c99 -g -Wall -Werror
LDFLAGS = -lOpenCL
# |
# \----------------------------------------------------------

# /----------------------------------------------------------
# | Compilation of matmul, matmul-blocked, matmul-blas
# +----------------------------------------------------------
# |

.PHONY:	all
all:	hello-gpu

# Compile a C version (using basic_dgemm.c, in this case):

hello-gpu: hello-gpu.o cl-helper.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Generic Rules
%.o:%.c
	$(CC) -c $(CFLAGS) $(OPTFLAGS) $<
# |
# \----------------------------------------------------------

# /----------------------------------------------------------
# | Clean-up rules
# +----------------------------------------------------------
# | Again, use these only if you understand what's going on.

.PHONY:	clean realclean
clean:
	rm -f hello-gpu *.o
# |
# \----------------------------------------------------------

