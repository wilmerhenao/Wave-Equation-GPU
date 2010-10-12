__kernel void fd_update(
    const float dt2_over_dx2,
    __global float *new_and_hist_u,
    __global const float *u,
    const unsigned dim_x,
    const unsigned dim_y,
    const unsigned points
)
{
  for (int k = 1; k <= points; ++k)
  {
    // + 1 to account for ghost cells
    const int i = get_global_id(1) + 1;
    const int j = get_global_id(0) + 1;

    unsigned base = i + dim_x*(j + dim_y * k);

    new_and_hist_u[base] =
      2 * u[base] - new_and_hist_u[base]
      + dt2_over_dx2 * (
          - 6*u[base]
          + u[base - 1]
          + u[base + 1]
          + u[base - dim_x]
          + u[base + dim_x]
          + u[base + dim_x*dim_y]
          + u[base - dim_x*dim_y]
          );
  }
}
