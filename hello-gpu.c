#include "cl-helper.h"




int main()
{
  // print_platforms_devices();

  cl_context ctx;
  cl_command_queue queue;
  create_context_on("NVIDIA", NULL, 0, &ctx, &queue);

  char *knl_text = read_file("vec-add.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "sum", NULL);
  free(knl_text);

  const size_t sz = 10000;
  float a[sz], b[sz], c[sz];

  for (size_t i = 0; i < sz; ++i)
  {
    a[i] = i;
    b[i] = 2*i;
  }

  cl_int status;
  cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
      sizeof(float) * sz, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(float) * sz, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_c = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(float) * sz, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sz * sizeof(float), a,
        0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sz * sizeof(float), b,
        0, NULL, NULL));

  SET_3_KERNEL_ARGS(knl, buf_a, buf_b, buf_c);
  size_t gdim[] = { sz };
  size_t ldim[] = { 1 };
  CALL_CL_GUARDED(clEnqueueNDRangeKernel,
      (queue, knl,
       /*dimensions*/ 1, NULL, gdim, ldim,
       0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueReadBuffer, (
        queue, buf_c, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sz * sizeof(float), c,
        0, NULL, NULL));

  for (size_t i = 0; i < sz; ++i)
    if (c[i] != 3*i)
      printf("BAD %ld!\n", i);

  CALL_CL_GUARDED(clReleaseMemObject, (buf_a));
  CALL_CL_GUARDED(clReleaseMemObject, (buf_b));
  CALL_CL_GUARDED(clReleaseMemObject, (buf_c));
  CALL_CL_GUARDED(clReleaseKernel, (knl));
  CALL_CL_GUARDED(clReleaseContext, (ctx));
}
