typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t __builtin_load(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t __builtin_block_idx(void);
uint32_t __builtin_thread_idx(void);
uint32_t __builtin_block_dim(void);
void __builtin_spawn(unsigned int, unsigned int,
                     void (*)(uint32_t),
                     uint32_t);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel()
{
    uint32_t p = __builtin_load(p_inp_values_p);
    for (uint32_t b = 0; b < 1; b++)
    {
        uint32_t i_0 = 0;
        uint32_t addr_0 = p + i_0 * 8;
        vec8_t a_0 = __builtin_vbroadcast(i_0);
        __builtin_vstore(addr_0, a_0);

        uint32_t i_1 = 1;
        uint32_t addr_1 = p + i_1 * 8;
        vec8_t a_1 = __builtin_vbroadcast(i_1);
        __builtin_vstore(addr_1, a_1);

        uint32_t i_2 = 2;
        uint32_t addr_2 = p + i_2 * 8;
        vec8_t a_2 = __builtin_vbroadcast(i_2);
        __builtin_vstore(addr_2, a_2);

        uint32_t i_3 = 3;
        uint32_t addr_3 = p + i_3 * 8;
        vec8_t a_3 = __builtin_vbroadcast(i_3);
        __builtin_vstore(addr_3, a_3);
    }
}
