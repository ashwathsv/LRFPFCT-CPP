#ifndef LRFPFCT_REACTION_2D_K_H_
#define LRFPFCT_REACTION_2D_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "LRFPFCT_EOS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
fct_react_convflux_x (int i, int j, int k,
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
                amrex::Array4<amrex::Real> const& uc,
                amrex::Array4<amrex::Real> const& utemp,
                amrex::Real dt,
                const int nsub,
                EOSParm const& eos_parm) noexcept
{
    using amrex::Real;

    Real rhoinv = Real(1.0) / uc(i,j,k,URHO);

    for(int ns = 0; ns < nsub; ++ns){
        Real omegarho = - uc(i,j,k,URHO) * eos_parm.pre_exp * uc(i,j,k,RHOY) 
                      * std::exp(-eos_parm.Ea_dim / (eos_parm.Ru * uc(i,j,k,UTEMP)));
        uc(i,j,k,UEDEN) = uc(i,j,k,UEDEN) - (dt * omegarho * eos_parm.q_dim);
        uc(i,j,k,URHOY) = uc(i,j,k,URHOY) + (dt * omegarho);

        if(uc(i,j,k,RHOY) * rhoinv > Real(1.0)) uc(i,j,k,URHOY) = uc(i,j,k,RHO);

        uc(i,j,k,UPRE) = (eos_parm.eos_gamma - Real(1.0)) * (uc(i,j,k,UEDEN)
                       - ( 0.5 * rhoinv * 
                         ( uc(i,j,k,UMX)*uc(i,j,k,UMX) + uc(i,j,k,UMY)*uc(i,j,k,UMY) )) );
        uc(i,j,k,UTEMP) = uc(i,j,k,UPRE) / (uc(i,j,k,URHO) * eos_parm.Rsp);
    }

    utemp(i,j,k,UEDEN) = uc(i,j,k,UEDEN);
    utemp(i,j,k,URHOY) = uc(i,j,k,URHOY);
}

#endif