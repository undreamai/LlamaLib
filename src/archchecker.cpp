#include "archchecker.h"

#include "cpu_x86.h"

using namespace FeatureDetector;

const bool has_avx()
{
    int32_t info[4];
    cpu_x86::cpuid(info, 0, 0);
    int nIds = info[0];

    if (nIds >= 0x00000001)
    {
        cpu_x86::cpuid(info, 0x00000001, 0);
        return (info[2] & ((int)1 << 28)) != 0;
    }
    return false;
}

const bool has_avx2()
{
    int32_t info[4];
    cpu_x86::cpuid(info, 0, 0);
    int nIds = info[0];

    if (nIds >= 0x00000007)
    {
        cpu_x86::cpuid(info, 0x00000007, 0);
        return (info[1] & ((int)1 << 5)) != 0;
    }
    return false;
}

const bool has_avx512()
{
    int32_t info[4];
    cpu_x86::cpuid(info, 0, 0);
    int nIds = info[0];

    if (nIds >= 0x00000007)
    {
        cpu_x86::cpuid(info, 0x00000007, 0);
        bool HW_AVX512_F = (info[1] & ((int)1 << 16)) != 0;
        bool HW_AVX512_BW = (info[1] & ((int)1 << 30)) != 0;
        return HW_AVX512_F && HW_AVX512_BW;
    }
    return false;
}

/*
int main(int argc, char ** argv) {
    std::cout<<has_avx()<<std::endl;
    std::cout<<has_avx2()<<std::endl;
    std::cout<<has_avx512()<<std::endl;
}
*/