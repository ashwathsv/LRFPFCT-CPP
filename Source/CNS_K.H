#ifndef CNS_K_H_
#define CNS_K_H_

#include "CNS_index_macros.H"
#include <AMReX_FArrayBox.H>
#include <limits>
#include <cmath>

#include "cns_prob.H"
#include "CNS_parm.H"

AMREX_GPU_HOST_DEVICE
inline
amrex::Real
cns_estdt (amrex::Box const& bx, amrex::Array4<Real const> const& state,
           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
           int ng,
           Parm const& parm) noexcept
{
    using amrex::Real;

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
    Real dt = std::numeric_limits<Real>::max();
#else
    Real dt = Real(1.e37);
#endif

#if AMREX_SPACEDIM==2
    for         (int k = lo.z; k <= hi.z; ++k) {
#else
    for         (int k = lo.z-ng; k <= hi.z+ng; ++k) {
#endif    
        for     (int j = lo.y-ng; j <= hi.y+ng; ++j) {
            for (int i = lo.x-ng; i <= hi.x+ng; ++i) {
                Real rho = state(i,j,k,URHO);
                Real mx  = state(i,j,k,UMX);
                Real my  = state(i,j,k,UMY);
                Real rhoinv = Real(1.0)/amrex::max(rho,parm.smallr);
                Real vx = mx*rhoinv;
                Real vy = my*rhoinv;
                Real p = state(i,j,k,UPRE);
                Real cs = std::sqrt(parm.eos_gamma*p*rhoinv);
                Real dtx = dx[0]/(amrex::Math::abs(vx)+cs);
                Real dty = dx[1]/(amrex::Math::abs(vy)+cs);
#if AMREX_SPACEDIM==2
                dt = amrex::min(dt,amrex::min(dtx,dty));
#else
                Real mz  = state(i,j,k,UMZ);
                Real vz = mz*rhoinv;
                Real dtz = dx[2]/(amrex::Math::abs(vz)+cs);
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
#endif
            }
        }
    }

    return dt;
}


AMREX_GPU_DEVICE
inline
void
cns_compute_temperature (int i, int j, int k, amrex::Array4<amrex::Real> const& u,
                         Parm const& parm) noexcept
{
    using amrex::Real;

    Real ux = u(i,j,k,UMX)/u(i,j,k,URHO);
    Real uy = u(i,j,k,UMY)/u(i,j,k,URHO);

    // Correction for negative densities
    
    if(u(i,j,k,URHO) < Real(0.0) && parm.do_minp == 1){
        u(i,j,k,URHO) = parm.minro;
        u(i,j,k,UMX) = u(i,j,k,URHO)*ux;
        u(i,j,k,UMY) = u(i,j,k,URHO)*uy;
    }

    Real rhoinv = Real(1.0)/u(i,j,k,URHO);
    Real mx = u(i,j,k,UMX);
    Real my = u(i,j,k,UMY);
#if AMREX_SPACEDIM==2
    Real kineng = Real(0.5) * rhoinv * (mx*mx+my*my);
    u(i,j,k,UEINT) = u(i,j,k,UEDEN) - Real(0.5) * rhoinv * (mx*mx+my*my);
#else
    Real mz = u(i,j,k,UMZ);
    Real kineng = Real(0.5) * rhoinv * (mx*mx+my*my+mz*mz);
    u(i,j,k,UEINT) = u(i,j,k,UEDEN) - Real(0.5) * rhoinv * (mx*mx+my*my+mz*mz);
#endif
    u(i,j,k,UTEMP) = rhoinv * u(i,j,k,UEINT) * (Real(1.0)/parm.cv);
    u(i,j,k,UPRE)  = (parm.eos_gamma-1.0) * ( u(i,j,k,UEDEN) - kineng );

    // Correction for negative pressures

    if(u(i,j,k,UPRE) < Real(0.0) && parm.do_minp == 1){
        u(i,j,k,UPRE)  = parm.minp;
        u(i,j,k,UEINT) = u(i,j,k,UPRE)/(parm.eos_gamma-Real(1.0));
        u(i,j,k,UEDEN) = u(i,j,k,UEINT) + kineng; 
        u(i,j,k,UTEMP) = rhoinv * u(i,j,k,UEINT) * (Real(1.0)/parm.cv);
    }
}

AMREX_GPU_HOST_DEVICE
inline
amrex::Real
cns_estdt_pureadv (amrex::Box const& bx, amrex::Array4<Real const> const& state,
           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
           Parm const& parm) noexcept
{
    using amrex::Real;

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
    Real dt = std::numeric_limits<Real>::max();
#else
    Real dt = Real(1.e37);
#endif

    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                Real rho = state(i,j,k,URHO);
                Real mx  = state(i,j,k,UMX);
                Real my  = state(i,j,k,UMY);
                Real rhoinv = Real(1.0)/amrex::max(rho,parm.smallr);
                Real vx = mx*rhoinv;
                Real vy = my*rhoinv;
                Real dtx = dx[0]/(amrex::Math::abs(vx));
                Real dty = dx[1]/(amrex::Math::abs(vy));
#if AMREX_SPACEDIM==2
                dt = amrex::min(dt,amrex::min(dtx,dty));
#else
                Real mz  = state(i,j,k,UMZ);
                Real vz = mz*rhoinv;
                Real dtz = dx[2]/(amrex::Math::abs(vz));
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
#endif
            }
        }
    }

    return dt;
}

#endif
