#ifndef LRFPFCT_DIFFUSION_2D_K_H_
#define LRFPFCT_DIFFUSION_2D_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "LRFPFCT_EOS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
physical_diffusion_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& fldx,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& uc,
                amrex::Array4<amrex::Real const> const& vel,
                amrex::Real dxinv, amrex::Real dyinv,
                EOSParm const& eos_parm) noexcept
{
    using amrex::Real;


    Real kappa  = eos_parm.kappa_0 * std::pow(uc(i,j,k,UTEMP),eos_parm.temp_exp);
    Real rhoD   = kappa / eos_parm.Le;
    Real mu = kappa * eos_parm.Pr;
    kappa = kappa * eos_parm.cp;

    fldx(i,j,k,URHO)  = Real(0.0);
    fldx(i,j,k,UMX)   = mu * ( (dxinv * Real(4.0/3.0) * (vel(i,j,k,0) - vel(i-1,j,k,0))) 
                      - (dyinv * Real(1.0/6.0) * (vel(i,j+1,k,1) - vel(i,j-1,k,1) 
                      + vel(i-1,j,k,1) - vel(i-1,j-1,k,1)) ) );
    fldx(i,j,k,UMY)   = mu * ( (Real(0.25) * dyinv * (vel(i,j+1,k,0) - vel(i,j-1,k,0) 
                      +          vel(i-1,j+1,k,0) - vel(i-1,j-1,k,0))) 
                      + (dxinv * ( vel(i,j,k,1) - vel(i-1,j,k,1) ) ) );
    fldx(i,j,k,UEDEN) = Real(0.5) * ( vel(i,j,k,0) + vel(i-1,j,k,0) ) * fldx(i,j,k,UMX)
                      + Real(0.5) * ( vel(i,j,k,1) + vel(i-1,j,k,1) ) * fldx(i,j,k,UMY)
                      + dxinv * kappa * ( uc(i,j,k,UTEMP) - uc(i-1,j,k,UTEMP) );
#ifdef LRFPFCT_REACTION
    fldx(i,j,k,URHOY) = rhoD * dxinv * 
                        ( (uc(i,j,k,URHOY)/uc(i,j,k,URHO)) - (uc(i-1,j,k,URHOY)/uc(i-1,j,k,URHO)) );  
#endif
}

AMREX_GPU_DEVICE
inline
void
physical_diffusion_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& fldy,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& uc,
                amrex::Array4<amrex::Real const> const& vel,
                amrex::Real dxinv, amrex::Real dyinv,
                EOSParm const& eos_parm) noexcept
{
    using amrex::Real;


    Real kappa  = eos_parm.kappa_0 * std::pow(uc(i,j,k,UTEMP),eos_parm.temp_exp);
    Real rhoD   = kappa / eos_parm.Le;
    Real mu = kappa * eos_parm.Pr;
    kappa = kappa * eos_parm.cp;

    fldy(i,j,k,URHO)  = Real(0.0);
    fldy(i,j,k,UMX)   = mu * ( (Real(0.25) * dxinv * (vel(i+1,j-1,k,1) - vel(i-1,j-1,k,1) 
                      +          vel(i+1,j,k,1) - vel(i-1,j,k,1))) 
                      + (dyinv * ( vel(i,j,k,0) - vel(i,j-1,k,0) ) ) );
    fldy(i,j,k,UMY)   = mu * ( (dxinv * Real(4.0/3.0) * (vel(i,j,k,1) - vel(i,j-1,k,1))) 
                      - (dyinv * Real(1.0/6.0) * (vel(i+1,j,k,0) - vel(i-1,j,k,0) 
                      + vel(i+1,j-1,k,0) - vel(i-1,j-1,k,0)) ) );
    fldy(i,j,k,UEDEN) = Real(0.5) * ( vel(i,j,k,0) + vel(i,j-1,k,0) ) * fldy(i,j,k,UMX)
                      + Real(0.5) * ( vel(i,j,k,1) + vel(i,j-1,k,1) ) * fldy(i,j,k,UMY)
                      + dyinv * kappa * ( uc(i,j,k,UTEMP) - uc(i,j-1,k,UTEMP) );
#ifdef LRFPFCT_REACTION
    fldy(i,j,k,URHOY) = rhoD * dyinv * 
                        ( (uc(i,j,k,URHOY)/uc(i,j,k,URHO)) - (uc(i,j-1,k,URHOY)/uc(i,j-1,k,URHO)) );  
#endif
}

#endif