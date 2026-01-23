#ifndef BFLOAT16_H
#define BFLOAT16_H

#include <stdint.h>
#include <string.h>

typedef struct {
    uint16_t v;
} bfloat16;

/* float -> bfloat16 (round-to-nearest-even) */
static inline bfloat16 float_to_bf16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint32_t lsb = (x >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;

    bfloat16 bf;
    bf.v = (uint16_t)((x + rounding_bias) >> 16);
    return bf;
}

/* bfloat16 -> float */
static inline float bf16_to_float(bfloat16 bf)
{
    uint32_t x = ((uint32_t)bf.v) << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

#endif