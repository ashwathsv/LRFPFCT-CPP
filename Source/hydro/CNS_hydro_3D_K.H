#ifndef CNS_HYDRO_3D_K_H_
#define CNS_HYDRO_3D_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
fct_con_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flcx,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& ubord) noexcept
{
    using amrex::Real;

    Real ux = Real(0.5)*(ubord(i-1,j,k,UMX)/ubord(i-1,j,k,URHO) 
            + ubord(i,j,k,UMX)/ubord(i,j,k,URHO));
    Real pint = Real(0.5)*(ubord(i-1,j,k,UPRE) + ubord(i,j,k,UPRE)); 
    flcx(i,j,k,URHO) = ux*Real(0.5)*(uold(i-1,j,k,URHO) + uold(i,j,k,URHO));
    flcx(i,j,k,UMX)  = ( pint + ux*Real(0.5)*(uold(i-1,j,k,UMX) + uold(i,j,k,UMX)) );
    flcx(i,j,k,UMY)  = ux*Real(0.5)*( uold(i-1,j,k,UMY) + uold(i,j,k,UMY) );
    flcx(i,j,k,UMZ)  = ux*Real(0.5)*( uold(i-1,j,k,UMZ) + uold(i,j,k,UMZ) );
    flcx(i,j,k,UEDEN) = ux*( pint + Real(0.5)*( uold(i-1,j,k,UEDEN) + uold(i,j,k,UEDEN)));
    // flcx(i,j,k,URHOY) = ux*Real(0.5)*(uold(i-1,j,k,URHOY) + uold(i,j,k,URHOY));
}

AMREX_GPU_DEVICE
inline
void
fct_con_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flcy,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& ubord) noexcept
{
    using amrex::Real;

    Real uy = Real(0.5)*(ubord(i,j-1,k,UMY)/ubord(i,j-1,k,URHO) 
            + ubord(i,j,k,UMY)/ubord(i,j,k,URHO));
    Real pint = Real(0.5)*(ubord(i,j-1,k,UPRE) + ubord(i,j,k,UPRE)); 
    flcy(i,j,k,URHO) = uy*Real(0.5)*( uold(i,j-1,k,URHO) + uold(i,j,k,URHO) );
    flcy(i,j,k,UMX)  = uy*Real(0.5)*( uold(i,j-1,k,UMX) + uold(i,j,k,UMX) );
    flcy(i,j,k,UMY)  = ( pint + uy*Real(0.5)*(uold(i,j-1,k,UMY) + uold(i,j,k,UMY)) );
    flcy(i,j,k,UMZ) = uy*Real(0.5)*( uold(i,j-1,k,UMZ) + uold(i,j,k,UMZ) );
    flcy(i,j,k,UEDEN) = uy*( pint + Real(0.5)*( uold(i,j-1,k,UEDEN) + uold(i,j,k,UEDEN)));
    // flcy(i,j,k,URHOY)  = uy*Real(0.5)*( uold(i,j-1,k,URHOY) + uold(i,j,k,URHOY) );
}

AMREX_GPU_DEVICE
inline
void
fct_con_flux_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flcz,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& ubord) noexcept
{
    using amrex::Real;

    Real uz = Real(0.5)*(ubord(i,j,k-1,UMZ)/ubord(i,j,k-1,URHO) 
            +            ubord(i,j,k,UMZ)/ubord(i,j,k,URHO));
    Real pint = Real(0.5)*(ubord(i,j,k-1,UPRE) + ubord(i,j,k,UPRE)); 
    flcz(i,j,k,URHO)  = uz*Real(0.5)*( uold(i,j,k-1,URHO) + uold(i,j,k,URHO) );
    flcz(i,j,k,UMX)   = uz*Real(0.5)*( uold(i,j,k-1,UMX) + uold(i,j,k,UMX) );
    flcz(i,j,k,UMY)   = uz*Real(0.5)*( uold(i,j,k-1,UMY) + uold(i,j,k,UMY) );
    flcz(i,j,k,UMZ)   = ( pint + uz*Real(0.5)*(uold(i,j,k-1,UMZ) + uold(i,j,k,UMZ)) );
    flcz(i,j,k,UEDEN) = uz*( pint + Real(0.5)*( uold(i,j,k-1,UEDEN) + uold(i,j,k,UEDEN)));
    // flcy(i,j,k,URHOY)  = uy*Real(0.5)*( uold(i,j-1,k,URHOY) + uold(i,j,k,URHOY) );
}

AMREX_GPU_DEVICE
inline
void
fct_diff_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& fldx,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& /*ubord*/,
                amrex::Real dxinv, amrex::Real dt) noexcept
{
    using amrex::Real;

    Real ux = Real(0.5)*(uold(i-1,j,k,UMX)/uold(i-1,j,k,URHO) 
            + uold(i,j,k,UMX)/uold(i,j,k,URHO));
    Real epsx = dt*dxinv*ux;
    Real nuxx = Real(1.0/12.0) + (Real(1.0/3.0)*epsx*epsx); 
    for(int n = URHO ; n <= UEDEN; ++n){
        fldx(i,j,k,n) = nuxx*(uold(i-1,j,k,n) - uold(i,j,k,n));
    }
    // fldx(i,j,k,UMX)  = nuxx*(uold(i-1,j,k,UMX)  - uold(i,j,k,UMX));
    // fldx(i,j,k,UMY)  = nuxx*(uold(i-1,j,k,UMY)  - uold(i,j,k,UMY));
    // fldx(i,j,k,UMZ)  = nuxx*(uold(i-1,j,k,UMZ) - uold(i,j,k,UMZ));
    // fldx(i,j,k,UEDEN) = nuxx*(uold(i-1,j,k,UEDEN) - uold(i,j,k,UEDEN));
}

AMREX_GPU_DEVICE
inline
void
fct_diff_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& fldy,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& /*ubord*/,
                amrex::Real dyinv, amrex::Real dt) noexcept
{
    using amrex::Real;

    Real uy = Real(0.5)*(uold(i,j-1,k,UMY)/uold(i,j-1,k,URHO) 
            + uold(i,j,k,UMY)/uold(i,j,k,URHO));
    Real epsy = dt*dyinv*uy;
    Real nuyy = Real(1.0/12.0) + (Real(1.0/3.0)*epsy*epsy); 
    for(int n = URHO; n <= UEDEN; ++n){
        fldy(i,j,k,n) = nuyy*( uold(i,j-1,k,n) - uold(i,j,k,n) );
    }
}

AMREX_GPU_DEVICE
inline
void
fct_diff_flux_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& fldz,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& /*ubord*/,
                amrex::Real dzinv, amrex::Real dt) noexcept
{
    using amrex::Real;

    Real uz = Real(0.5)*(uold(i,j,k-1,UMZ)/uold(i,j,k-1,URHO) 
            + uold(i,j,k,UMZ)/uold(i,j,k,URHO));
    Real epsz = dt*dzinv*uz;
    Real nuzz = Real(1.0/12.0) + (Real(1.0/3.0)*epsz*epsz); 
    for(int n = URHO; n <= UEDEN; ++n){
        fldz(i,j,k,n)  = nuzz*( uold(i,j,k-1,n) - uold(i,j,k,n) );
    }
}

AMREX_GPU_DEVICE
inline
void
fct_ad_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& /*ubord*/,
                amrex::Array4<amrex::Real> const& ucx,
                amrex::Real dxinv, amrex::Real dt,
                amrex::Real diff1) noexcept
{
    using amrex::Real;

    Real ux = Real(0.5)*(uold(i-1,j,k,UMX)/uold(i-1,j,k,URHO) 
            + uold(i,j,k,UMX)/uold(i,j,k,URHO));
    Real epsx = dt*dxinv*ux;
    Real muxx = (Real(1.0/6.0) - (Real(1.0/6.0)*epsx*epsx)); 
    for(int n = URHO; n <= UEDEN; ++n){
        flax(i,j,k,n) = diff1*(muxx*(ucx(i,j,k,n) -  ucx(i-1,j,k,n)) 
                         - Real(1.0/12.0)*(uold(i,j,k,n) - uold(i-1,j,k,n)));        
    }
    // flax(i,j,k,URHO) = diff1*(muxx*(ucx(i,j,k,URHO) -  ucx(i-1,j,k,URHO)) 
    //                  - Real(1.0/12.0)*(uold(i,j,k,URHO) - uold(i-1,j,k,URHO)));
}

AMREX_GPU_DEVICE
inline
void
fct_ad_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& /*ubord*/,
                amrex::Array4<amrex::Real> const& ucy,
                amrex::Real dyinv, amrex::Real dt,
                amrex::Real diff1) noexcept
{
    using amrex::Real;

    Real uy = Real(0.5)*(uold(i,j-1,k,UMY)/uold(i,j-1,k,URHO) 
            + uold(i,j,k,UMY)/uold(i,j,k,URHO));
    Real epsy = dt*dyinv*uy;
    Real muyy = (Real(1.0/6.0) - (Real(1.0/6.0)*epsy*epsy)); 
    for(int n = URHO; n <= UEDEN; ++n){
        flay(i,j,k,n) = diff1*(muyy*(ucy(i,j,k,n) -  ucy(i,j-1,k,n)) 
                          - Real(1.0/12.0)*(uold(i,j,k,n) - uold(i,j-1,k,n)));
    }
    // flay(i,j,k,UMX)  = muyy*(ucy(i,j,k,UMX) -  ucy(i,j-1,k,UMX)) 
    //                  - Real(1.0/12.0)*(uold(i,j,k,UMX) - uold(i,j-1,k,UMX));
}

AMREX_GPU_DEVICE
inline
void
fct_ad_flux_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real const> const& uold,
                amrex::Array4<amrex::Real const> const& /*ubord*/,
                amrex::Array4<amrex::Real> const& ucz,
                amrex::Real dzinv, amrex::Real dt,
                amrex::Real diff1) noexcept
{
    using amrex::Real;

    Real uz = Real(0.5)*(uold(i,j,k-1,UMZ)/uold(i,j,k-1,URHO) 
            + uold(i,j,k,UMZ)/uold(i,j,k,URHO));
    Real epsz = dt*dzinv*uz;
    Real muzz = (Real(1.0/6.0) - (Real(1.0/6.0)*epsz*epsz)); 
    for(int n = URHO; n <= UEDEN; ++n){
        flaz(i,j,k,n) = diff1*(muzz*(ucz(i,j,k,n) -  ucz(i,j,k-1,n)) 
                      - Real(1.0/12.0)*(uold(i,j,k,n) - uold(i,j,k-1,n)));        
    }    
    // flaz(i,j,k,UMX)  = muzz*(ucz(i,j,k,UMX) -  ucz(i,j,k-1,UMX)) 
    //                  - Real(1.0/12.0)*(uold(i,j,k,UMX) - uold(i,j,k-1,UMX));
}

AMREX_GPU_DEVICE
inline
void
fct_prelimit_ad_flux_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& udx,
                int ilo, int jlo, int klo,
                int ihi, int jhi, int khi) noexcept
{
    using amrex::Real;

    Real fltmp = amrex::Math::abs(flax(i,j,k,n));
    Real sgn = amrex::Math::copysign(1.0, udx(i,j,k,n) - udx(i-1,j,k,n));
    flax(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udx(i-1,j,k,n) - udx(i-2,j,k,n)),
                                   sgn*(udx(i+1,j,k,n) - udx(i,j,k,n))  ));      
}

AMREX_GPU_DEVICE
inline
void
fct_prelimit_ad_flux_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& udy,
                int ilo, int jlo, int klo,
                int ihi, int jhi, int khi) noexcept
{
    using amrex::Real;

    Real fltmp = amrex::Math::abs(flay(i,j,k,n));
    Real sgn = amrex::Math::copysign(1.0, udy(i,j,k,n) - udy(i,j-1,k,n));
    flay(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udy(i,j-1,k,n) - udy(i,j-2,k,n)),
                                   sgn*(udy(i,j+1,k,n) - udy(i,j,k,n))  ));
}

AMREX_GPU_DEVICE
inline
void
fct_prelimit_ad_flux_z (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& udz,
                int ilo, int jlo, int klo,
                int ihi, int jhi, int khi) noexcept
{
    using amrex::Real;

    Real fltmp = amrex::Math::abs(flaz(i,j,k,n));
    Real sgn = amrex::Math::copysign(1.0, udz(i,j,k,n) - udz(i,j,k-1,n));
    flaz(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udz(i,j,k-1,n) - udz(i,j,k-2,n)),
                                   sgn*(udz(i,j,k+1,n) - udz(i,j,k,n))  ));
}

AMREX_GPU_DEVICE
inline
void
fct_compute_frac_fluxes (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                amrex::Array4<amrex::Real> const& utmp) noexcept
{
    using amrex::Real;

    Real fintmp  = amrex::max(flax(i,j,k,n),Real(0.0)) - amrex::min(flax(i+1,j,k,n),Real(0.0))
                 + amrex::max(flay(i,j,k,n),Real(0.0)) - amrex::min(flay(i,j+1,k,n),Real(0.0));

    Real fouttmp = amrex::max(flax(i+1,j,k,n),Real(0.0)) - amrex::min(flax(i,j,k,n),Real(0.0))
                 + amrex::max(flay(i,j+1,k,n),Real(0.0)) - amrex::min(flay(i,j,k,n),Real(0.0));

    Real umin    = amrex::min(utmp(i-1,j,k,n),utmp(i,j-1,k,n),utmp(i,j,k,n),
                              utmp(i+1,j,k,n),utmp(i,j+1,k,n));
    Real umax    = amrex::max(utmp(i-1,j,k,n),utmp(i,j-1,k,n),utmp(i,j,k,n),
                              utmp(i+1,j,k,n),utmp(i,j+1,k,n));

    fintmp  += amrex::max(flaz(i,j,k,n),Real(0.0)) - amrex::min(flaz(i,j,k+1,n),Real(0.0));
    fouttmp += amrex::max(flaz(i,j,k+1,n),Real(0.0)) - amrex::min(flaz(i,j,k,n),Real(0.0));
    umin = amrex::min(umin,utmp(i,j,k-1,n),utmp(i,j,k+1,n));
    umax = amrex::max(umax,utmp(i,j,k-1,n),utmp(i,j,k+1,n)); 

    frin(i,j,k,n)  = (umax - utmp(i,j,k,n))/(Real(1E-16) + fintmp);
    frout(i,j,k,n) = (utmp(i,j,k,n) - umin)/(Real(1E-16) + fouttmp); 

}

AMREX_GPU_DEVICE
inline
void
fct_correct_fluxes_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout) noexcept
{
    using amrex::Real;

    Real fltmp = flax(i,j,k,n);
    if(fltmp >= Real(0.0))
        flax(i,j,k,n) = fltmp*amrex::min(frout(i-1,j,k,n),frin(i,j,k,n),Real(1.0));
    else
        flax(i,j,k,n) = fltmp*amrex::min(frin(i-1,j,k,n),frout(i,j,k,n),Real(1.0));
    
}

AMREX_GPU_DEVICE
inline
void
fct_correct_fluxes_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout) noexcept
{
    using amrex::Real;

    Real fltmp = flay(i,j,k,n);
    if(fltmp >= Real(0.0))
        flay(i,j,k,n) = fltmp*amrex::min(frout(i,j-1,k,n),frin(i,j,k,n),Real(1.0));
    else
        flay(i,j,k,n) = fltmp*amrex::min(frin(i,j-1,k,n),frout(i,j,k,n),Real(1.0));
    
}

AMREX_GPU_DEVICE
inline
void
fct_correct_fluxes_z (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout) noexcept
{
    using amrex::Real;

    Real fltmp = flaz(i,j,k,n);
    if(fltmp >= Real(0.0))
        flaz(i,j,k,n) = fltmp*amrex::min(frout(i,j,k-1,n),frin(i,j,k,n),Real(1.0));
    else
        flaz(i,j,k,n) = fltmp*amrex::min(frin(i,j,k-1,n),frout(i,j,k,n),Real(1.0));
    
}

#endif