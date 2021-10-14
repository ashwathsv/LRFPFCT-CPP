#ifndef LRFPFCT_REACTION_2D_K_H_
#define LRFPFCT_REACTION_2D_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "LRFPFCT_EOS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

#ifdef LRFPFCT_REACTION
AMREX_GPU_DEVICE
inline
void
fct_reaction_terms (int i, int j, int k,
                amrex::Array4<amrex::Real> const& omegarho,
                amrex::Array4<amrex::Real> const& u0,
                amrex::Real dt,
                const int nsub,
                EOSParm const& eos_parm) noexcept
{
    using amrex::Real;

    omegarho(i,j,k,0) = -u0(i,j,k,URHO) * eos_parm.pre_exp * u0(i,j,k,URHOY)
                      * std::exp(-eos_parm.Ea_dim / (eos_parm.Ru * u0(i,j,k,UTEMP)));
}

AMREX_GPU_DEVICE
inline
void
fct_add_reaction_source_terms (int i, int j, int k,
                amrex::Array4<amrex::Real> const& udx,
                amrex::Array4<amrex::Real> const& udy,
                amrex::Array4<amrex::Real> const& dudt,
                amrex::Array4<amrex::Real> const& omegarho,
                amrex::Array4<amrex::Real> const& ud,
                EOSParm const& eos_parm,
                amrex::Real dt) noexcept
{
    using amrex::Real;

    Real omegaq = dt * omegarho(i,j,k,0) * eos_parm.q_dim;

    udx(i,j,k,UEDEN) = udx(i,j,k,UEDEN) - omegaq;
    udx(i,j,k,URHOY) = amrex::max(0.0,amrex::min((udx(i,j,k,URHOY) + (dt*omegarho(i,j,k,0))), udx(i,j,k,URHO)));
    
    udy(i,j,k,UEDEN) = udy(i,j,k,UEDEN) - omegaq;
    udy(i,j,k,URHOY) = amrex::max(0.0,amrex::min((udy(i,j,k,URHOY) + (dt*omegarho(i,j,k,0))), udy(i,j,k,URHO)));

    ud(i,j,k,UEDEN) = ud(i,j,k,UEDEN) - omegaq;
    ud(i,j,k,URHOY) = amrex::max(0.0,amrex::min((ud(i,j,k,URHOY)   + (dt*omegarho(i,j,k,0))), ud(i,j,k,URHO)));

    dudt(i,j,k,UEDEN) = dudt(i,j,k,UEDEN) - omegaq;
    dudt(i,j,k,URHOY) = dudt(i,j,k,URHOY) + (dt*omegarho(i,j,k,0)); 
}
#endif

#endif