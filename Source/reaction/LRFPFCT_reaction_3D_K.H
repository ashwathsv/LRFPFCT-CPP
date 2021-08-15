#ifndef LRFPFCT_REACTION_3D_K_H_
#define LRFPFCT_REACTION_3D_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
fct_react_convflux_x  (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flcx,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& ubord) noexcept
{
    using amrex::Real;

    Real ux = Real(0.5)*(ubord(i-1,j,k,UMX)/ubord(i-1,j,k,URHO) 
            + ubord(i,j,k,UMX)/ubord(i,j,k,URHO));
    flcx(i,j,k,URHOY) = ux*Real(0.5)*(uold(i-1,j,k,URHOY) + uold(i,j,k,URHOY));
}

AMREX_GPU_DEVICE
inline
void
fct_react_convflux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flcy,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& ubord) noexcept
{
    using amrex::Real;

    Real uy = Real(0.5)*(ubord(i,j-1,k,UMY)/ubord(i,j-1,k,URHO) 
            + ubord(i,j,k,UMY)/ubord(i,j,k,URHO));
    flcy(i,j,k,URHOY)  = uy*Real(0.5)*( uold(i,j-1,k,URHOY) + uold(i,j,k,URHOY) );
}

AMREX_GPU_DEVICE
inline
void
fct_react_convflux_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flcz,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& ubord) noexcept
{
    using amrex::Real;

    Real uz = Real(0.5)*(ubord(i,j,k-1,UMZ)/ubord(i,j,k-1,URHO) 
            +            ubord(i,j,k,UMZ)/ubord(i,j,k,URHO));
    flcz(i,j,k,URHOY)  = uz*Real(0.5)*( uold(i,j,k-1,URHOY) + uold(i,j,k,URHOY) );
}

AMREX_GPU_DEVICE
inline
void
fct_reaction_rates (int i, int j, int k,
                amrex::Array4<amrex::Real> const& omega,
                amrex::Array4<amrex::Real const> const& uold,
                EOSParm const& eos_parm) noexcept
{
    using amrex::Real;

    omega(i,j,k,0) = -eos_parm.pre_exp * uold(i,j,k,URHOY) 
                   * std::exp(-eos_parm.Ea_dim / (eos_parm.Ru * uold(i,j,k,UTEMP)));
}

AMREX_GPU_DEVICE
inline
void
fct_add_reaction_source (int i, int j, int k,
                amrex::Array4<amrex::Real> const& omega,
                amrex::Array4<amrex::Real> const& ucx,
                amrex::Array4<amrex::Real> const& ucy,
                amrex::Array4<amrex::Real> const& ucz,
                amrex::Array4<amrex::Real> const& utemp,
                amrex::Array4<amrex::Real> const& unew,
                EOSParm const& eos_parm) noexcept
{
    using amrex::Real;

    Real omegarho = omega(i,j,k,0) * uold(i,j,k,URHO);
    ucx(i,j,k,UEDEN) = ucx(i,j,k,UEDEN) - (omegarho * eos_parm.q_dim);
    ucx(i,j,k,URHOY) = ucx(i,j,k,URHOY) + omegarho;

    ucy(i,j,k,UEDEN) = ucy(i,j,k,UEDEN) - (omegarho * eos_parm.q_dim);
    ucy(i,j,k,URHOY) = ucy(i,j,k,URHOY) + omegarho;

    ucz(i,j,k,UEDEN) = ucz(i,j,k,UEDEN) - (omegarho * eos_parm.q_dim);
    ucz(i,j,k,URHOY) = ucz(i,j,k,URHOY) + omegarho;

    utemp(i,j,k,UEDEN) = utemp(i,j,k,UEDEN) - (omegarho * eos_parm.q_dim);
    utemp(i,j,k,URHOY) = utemp(i,j,k,URHOY) + omegarho;

    unew(i,j,k,UEDEN) = utemp(i,j,k,UEDEN);
    unew(i,j,k,URHOY) = utemp(i,j,k,URHOY);
}

#endif