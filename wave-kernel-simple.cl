__kernel void fd_update(
    const float dt2_over_dx2,
    __global float *new_and_hist_u,
    __global const float *u,
    const unsigned dim_x,
    const unsigned dim_y,
    const unsigned points
)
{
  for (int k = 2; k < points; ++k)
  {
    // + 2 to account for ghost cells
    const int i = get_global_id(1) + 2;
    const int j = get_global_id(0) + 2;

    unsigned base = i + dim_x*(j + dim_y * k);

    new_and_hist_u[base] =
      2 * u[base] - new_and_hist_u[base]
      + (1.0/12.0) *dt2_over_dx2 * (
          - 90*u[base]
          + 16.0 * u[base - 1]
          + 16.0 * u[base + 1]
          + 16.0 * u[base - dim_x]
          + 16.0 * u[base + dim_x]
          + 16.0 * u[base + dim_x*dim_y]
          + 16.0 * u[base - dim_x*dim_y]
          - 1.0 * u[base - 2]
          - 1.0 * u[base + 2]
          - 1.0 * u[base - 2*dim_x]
          - 1.0 * u[base + 2*dim_x]
          - 1.0 * u[base + 2*dim_x*dim_y]
          - 1.0 * u[base - 2*dim_x*dim_y]
          );
  }
}
