#include "cl-helper.h"
#include <math.h>

#define DO_TIMING

int main()
{
#ifdef DO_TIMING
  double gbytes_accessed = 0;
  double mflops = 0;
  double mcells = 0;
  double seconds_taken = 0;
#endif

  print_platforms_devices();

  cl_context ctx;
  cl_command_queue queue;
  create_context_on("NVIDIA", NULL, 0, &ctx, &queue,
#ifdef DO_TIMING
      1
#else
      0
#endif
      );

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("wave-kernel-simple.cl");
  cl_kernel wave_knl = kernel_from_string(ctx, knl_text, "fd_update", NULL);
  free(knl_text);

  knl_text = read_file("source-term.cl");
  cl_kernel source_knl = kernel_from_string(ctx, knl_text, "add_source_term", NULL);
  free(knl_text);

  // --------------------------------------------------------------------------
  // set up grid
  // --------------------------------------------------------------------------
  const unsigned points = 64;

  const float minus_bdry = -1, plus_bdry = 1;

  // We're dividing into (points-1) intervals.
  float dx = (plus_bdry-minus_bdry)/(points+1);
  float dt = 0.5*dx;
  float dt2_over_dx2 = dt*dt / (dx*dx);

  const float final_time = 200;

  // This might run a little short, which is ok in our case.
  unsigned step_count = final_time/dt;
  printf("will take %d steps.\n", step_count);

  // --------------------------------------------------------------------------
  // allocate and initialize CPU memory
  // --------------------------------------------------------------------------

  // add 2 in each dimension for boundary condition data
  const size_t field_size = (points+2)*(points+2)*(points+2);
  float *host_buf = malloc(field_size*sizeof(float));
  CHECK_SYS_ERROR(!host_buf, "allocating host_buf");

  for (size_t i = 0; i < field_size; ++i)
    host_buf[i] = 0;

  // --------------------------------------------------------------------------
  // allocate GPU memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem dev_buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
      sizeof(float) * field_size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem dev_buf_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(float) * field_size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // --------------------------------------------------------------------------
  // transfer to GPU
  // --------------------------------------------------------------------------
  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, dev_buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
        field_size * sizeof(float), host_buf,
        0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, dev_buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
        field_size * sizeof(float), host_buf,
        0, NULL, NULL));

  // --------------------------------------------------------------------------
  // time step loop
  // --------------------------------------------------------------------------
  cl_mem cur_u = dev_buf_a, hist_u = dev_buf_b;

  unsigned dim_x = points + 2;
  unsigned dim_y = points + 2;

  for (unsigned step = 0; step < step_count; ++step)
  {
    float t = step * dt;
    if (step % 100 == 0)
    {
      printf("step %d\n", step);
      CALL_CL_GUARDED(clFinish, (queue));
    }

    // visualize, if necessary
    if (step % (points / 8) == 0 && t > 0.6)
    {
      CALL_CL_GUARDED(clEnqueueReadBuffer, (
            queue, cur_u, /*blocking*/ CL_TRUE, /*offset*/ 0,
            field_size * sizeof(float), host_buf,
            0, NULL, NULL));

      for (size_t i = 0; i < field_size; ++i)
        if (isnan(host_buf[i]))
        {
          fputs("nan encountered, aborting\n", stderr);
          goto cleanup;
        }

      char fnbuf[100];
      sprintf(fnbuf, "wave-%05d.bov", step);

      FILE *bov_header = fopen(fnbuf, "w");
      CHECK_SYS_ERROR(!bov_header, "opening vis header");

      sprintf(fnbuf, "wave-%05d.dat", step);
      fprintf(bov_header, "TIME: %g\n", t);
      fprintf(bov_header, "DATA_FILE: %s\n", fnbuf);
      fprintf(bov_header, "DATA_SIZE: %d %d %d\n", points+2, points+2, points+2);
      fputs("DATA_FORMAT: FLOAT\n", bov_header);
      fputs("VARIABLE: solution\n", bov_header);
      fputs("DATA_ENDIAN: LITTLE\n", bov_header);
      fputs("CENTERING: nodal\n", bov_header);
      fprintf(bov_header, "BRICK_ORIGIN: %g %g %g\n", minus_bdry, minus_bdry, minus_bdry);
      fprintf(bov_header, "BRICK_SIZE: %g %g %g\n", 
          plus_bdry-minus_bdry,
          plus_bdry-minus_bdry,
          plus_bdry-minus_bdry);
      fclose(bov_header);

      FILE *bov_data = fopen(fnbuf, "wb");
      CHECK_SYS_ERROR(!bov_header, "opening vis output");
      fwrite((void *)host_buf, sizeof(float), field_size, bov_data);
      fclose(bov_data);
    }

    {
      // invoke wave kernel
      size_t gdim[] = { points, points };
      size_t ldim[] = { 16, 16 };

      SET_6_KERNEL_ARGS(wave_knl, dt2_over_dx2, hist_u, cur_u, dim_x, dim_y, points);

#ifdef DO_TIMING
      cl_event evt;
      cl_event *evt_ptr = &evt;
#else
      cl_event *evt_ptr = NULL;
#endif
      // this is where you run the whole enchilada
      CALL_CL_GUARDED(clEnqueueNDRangeKernel,
          (queue, wave_knl,
           /*dimensions*/ 2, NULL, gdim, ldim,
           0, NULL, evt_ptr));

#ifdef DO_TIMING
      // If timing is enabled, this wait can mean a significant performance hit.
      CALL_CL_GUARDED(clWaitForEvents, (1, &evt));

      gbytes_accessed += 1e-9*(sizeof(float) * field_size * 10);
      mflops += 2.0 * points * 23 / 1.0e3; // I count 23 operations inside
      mcells += 6.0 * points / 1.0e3;
      cl_ulong start, end;
      CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, 
            CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
      CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, 
            CL_PROFILING_COMMAND_END, sizeof(start), &end, NULL));

      seconds_taken += 1e-9*(end-start);

      CALL_CL_GUARDED(clReleaseEvent, (evt));
#endif
    }

    {
      // After we have solved the stencil we want to add the term "f"
      // invoke source term kernel
      size_t gdim[] = { points, points, points };
      size_t ldim[] = { 32, 32 , 32};

      //unsigned base = (points/4) + dim_x*((points/5) + dim_y * (points/6));
      //float value = dt*dt*sin(20*t);
      SET_7_KERNEL_ARGS(source_knl, hist_u, points, dt, t, dim_x, dim_y, dx);
    #ifdef DO_TIMING
          cl_event evt;
          cl_event *evt_ptr = &evt;
    #else
          cl_event *evt_ptr = NULL;
    #endif
      CALL_CL_GUARDED(clEnqueueNDRangeKernel,
          (queue, source_knl,
           /*dimensions*/ 1, NULL, gdim, ldim,
           0, NULL, evt_ptr));

     #ifdef DO_TIMING
      // timing for the "f" part... which didn't exist before
      // If timing is enabled, this wait can mean a significant performance hit.
         CALL_CL_GUARDED(clWaitForEvents, (1, &evt));

         gbytes_accessed += 1e-9*(sizeof(float) * field_size * 10);
         mflops += 2.0 * points * 27 / 1.0e3; // I count 23 operations inside
         mcells += points * 6.0 /1.0e3;
         cl_ulong start, end;
         CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, 
               CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
         CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, 
               CL_PROFILING_COMMAND_END, sizeof(start), &end, NULL));

         seconds_taken += 1e-9*(end-start);

         CALL_CL_GUARDED(clReleaseEvent, (evt));
     #endif

    }

    // swap buffers
    cl_mem tmp = cur_u;
    cur_u = hist_u;
    hist_u = tmp;
  } // this is where the step ends

#ifdef DO_TIMING
  printf("MCells/sec: %g\n", mcells/seconds_taken);
  printf("MFlops/sec: %g\n", mflops/seconds_taken);
  printf("GBytes/sec: %g\n", gbytes_accessed/seconds_taken);
#endif

cleanup:
  CALL_CL_GUARDED(clFinish, (queue));

  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_a));
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_b));
  CALL_CL_GUARDED(clReleaseKernel, (wave_knl));
  CALL_CL_GUARDED(clReleaseKernel, (source_knl));
  CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
  CALL_CL_GUARDED(clReleaseContext, (ctx));

  free(host_buf);
}
