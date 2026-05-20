#include "archchecker_arm.h"

#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>

const bool has_dotprod()
{
    // HWCAP_ASIMDDP: sdot/udot (ARMv8.2-a dotprod extension, all Cortex-A76+ / 2018+)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
}

const bool has_i8mm()
{
    // HWCAP2_I8MM: int8 matrix multiply (ARMv8.6-a, Cortex-A510 / 2021+)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
}

#else

const bool has_dotprod() { return false; }
const bool has_i8mm() { return false; }

#endif
