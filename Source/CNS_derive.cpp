#include "CNS_derive.H"
#include "CNS.H"
#include "CNS_parm.H"

using namespace amrex;

// void cns_derpres (const Box& bx, FArrayBox& pfab, int dcomp, int /*ncomp*/,
//                   const FArrayBox& rhoefab, const Geometry& /*geomdata*/,
//                   Real /*time*/, const int* /*bcrec*/, int /*level*/)
// {
//     auto const rhoe = rhoefab.array();
//     auto       p    = pfab.array();
//     Parm const* parm = CNS::d_parm;
//     amrex::ParallelFor(bx,
//     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//     {
//         p(i,j,k,dcomp) = (parm->eos_gamma-1.)*rhoe(i,j,k);
//     });
// }

void cns_dervel (const Box& bx, FArrayBox& velfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       vel = velfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,dcomp) = dat(i,j,k,1)/dat(i,j,k,0);
    });
}

void cns_dermac (const Box& bx, FArrayBox& macfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       mac = macfab.array();
    Parm const* parm = CNS::d_parm;
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real velmod = std::sqrt( (dat(i,j,k,1)*dat(i,j,k,1) + dat(i,j,k,2)*dat(i,j,k,2)) )/dat(i,j,k,0);
        mac(i,j,k,dcomp) = velmod/(std::sqrt(parm->eos_gamma*dat(i,j,k,3)/dat(i,j,k,0)));
        // if(fabs( (dat(i,j,k,3)/parm->minp) - Real(1.0) ) <= Real(0.01) ) mac(i,j,k,dcomp) = Real(0.0);
    });
}

void cns_dermassfrac (const Box& bx, FArrayBox& yfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       yarr = yfab.array();
    Parm const* parm = CNS::d_parm;
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        yarr(i,j,k,dcomp) = dat(i,j,k,1)/dat(i,j,k,0);
    });
}
