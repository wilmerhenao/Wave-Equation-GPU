__kernel void add_source_term(
    __global float *u,
    const unsigned base,
    const float value
)
{
  u[base] += value;
}
