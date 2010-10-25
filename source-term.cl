#define BLOCK_SIZE 16
#define A_BLOCK_STRIDE (BLOCK_SIZE * dim_x)
#define A_T_BLOCK_STRIDE (BLOCK_SIZE * dim_y)

__kernel void add_source_term(
    __global float *u,
    const unsigned points,
    const float dt,
    const float t,
    const unsigned dim_x,
    const unsigned dim_y,
    const float dx
)
{
  __local float u_local[BLOCK_SIZE][BLOCK_SIZE];
  int base_idx_a =
    get_group_id(0) * BLOCK_SIZE +
    get_group_id(1) * A_BLOCK_STRIDE;
  int glob_idx_a =
    base_idx_a + get_local_id(0)
    + dim_x * get_local_id(1);

    __local float x, y, z;

    const int i = get_global_id(1) + 1;
    const int j = get_global_id(0) + 1;
    const int k = base_idx_a;
    
    x = -1 + (float) i * dx;
    y = -1 + (float) j * dx;
    z = -1 + (float) k * dx;
    unsigned base = i + dim_x * (j + dim_y * k);
    
    u[base] = dt * dt * exp(-1600 * ((x - 0.5) * (x - 0.5) + (y - 0.2) * (y - 0.2) + (z - 0.3) * (z - 0.3) ));

}
