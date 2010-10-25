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
  
  for (int k = 1; k <= points; ++k){
    __local float x, y, z;

    const int i = get_global_id(1) + 1;
    const int j = get_global_id(0) + 1;
    
    x = -1 + (float) i * dx;
    y = -1 + (float) j * dx;
    z = -1 + (float) k * dx;
    unsigned base = i + dim_x * (j + dim_y * k);
    
    u[base] = dt * dt * exp(-1600 * ((x - 0.5) * (x - 0.5) + (y - 0.2) * (y - 0.2) + (z - 0.3) * (z - 0.3) ));

  }
}
