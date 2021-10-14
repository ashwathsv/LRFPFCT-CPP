#ifndef CNS_HYDRO_2D_TMP_K_H_
#define CNS_HYDRO_2D_TMP_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
fct_con_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& vx,
                amrex::Array4<amrex::Real> const& flcx,
                amrex::Array4<amrex::Real> const& u0) noexcept
{
    using amrex::Real;

    Real ux    = Real(0.5)*( vx(i,j,k,0) + vx(i-1,j,k,0) );
    Real pint  = Real(0.5)*( u0(i,j,k,UPRE) + u0(i-1,j,k,UPRE) ); 
    Real puint = Real(0.5)*( (u0(i,j,k,UPRE)*vx(i,j,k,0)) + (u0(i-1,j,k,UPRE)*vx(i-1,j,k,0)) ); 
    flcx(i,j,k,URHO)  = ux*Real(0.5)*( u0(i,j,k,URHO)  + u0(i-1,j,k,URHO));
    flcx(i,j,k,UMX)   = ux*Real(0.5)*( u0(i,j,k,UMX)   + u0(i-1,j,k,UMX) ) + pint;
    flcx(i,j,k,UMY)   = ux*Real(0.5)*( u0(i,j,k,UMY)   + u0(i-1,j,k,UMY) );
    flcx(i,j,k,UEDEN) = ux*Real(0.5)*( u0(i,j,k,UEDEN) + u0(i-1,j,k,UEDEN) ) + puint;
#ifdef LRFPFCT_REACTION
    flcx(i,j,k,URHOY) = ux*Real(0.5)*( u0(i,j,k,URHOY) + u0(i-1,j,k,URHOY) );
#endif
}

AMREX_GPU_DEVICE
inline
void
fct_con_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& vy,
                amrex::Array4<amrex::Real> const& flcy,
                amrex::Array4<amrex::Real> const& u0) noexcept
{
    using amrex::Real;

    const Real half = Real(0.5);
    Real uy    = half*( vy(i,j,k,0) + vy(i,j-1,k,0) );
    Real pint  = half*( u0(i,j,k,UPRE) + u0(i,j-1,k,UPRE) ); 
    Real pvint = half*( (u0(i,j,k,UPRE)*vy(i,j,k,0)) + (u0(i,j-1,k,UPRE)*vy(i,j-1,k,0)) );
    flcy(i,j,k,URHO)  = uy*half*( u0(i,j,k,URHO)   + u0(i,j-1,k,URHO)  );
    flcy(i,j,k,UMX)   = uy*half*( u0(i,j,k,UMX)    + u0(i,j-1,k,UMX)   );
    flcy(i,j,k,UMY)   = (uy*half*(u0(i,j,k,UMY)    + u0(i,j-1,k,UMY))  ) + pint;
    flcy(i,j,k,UEDEN) = uy*( half*(u0(i,j,k,UEDEN) + u0(i,j-1,k,UEDEN))) + pvint;
#ifdef LRFPFCT_REACTION
    flcy(i,j,k,URHOY) = uy*( half*(u0(i,j,k,URHOY) + u0(i,j-1,k,URHOY)) );
#endif
}

AMREX_GPU_DEVICE
inline
void
fct_diff_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& vx,
                amrex::Array4<amrex::Real> const& fldx,
                amrex::Array4<amrex::Real const> const& u0,
                amrex::Real dxinv, amrex::Real dt,
                const int conscomp,
                const amrex::Real nudiff) noexcept
{
    using amrex::Real;

    const Real half = Real(0.5);
    Real ux = half*( vx(i,j,k,0) + vx(i-1,j,k,0) );
    Real epsx = dt*dxinv*ux;
    Real nuxx = nudiff + (Real(1.0/3.0)*epsx*epsx); 
    for(int n = URHO; n < conscomp; ++n){
        fldx(i,j,k,n)  = nuxx*(u0(i,j,k,n) - u0(i-1,j,k,n));
    }
}

AMREX_GPU_DEVICE
inline
void
fct_diff_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& vy,
                amrex::Array4<amrex::Real> const& fldy,
                amrex::Array4<amrex::Real const> const& u0,
                amrex::Real dyinv, amrex::Real dt,
                const int conscomp,
                const amrex::Real nudiff) noexcept
{
    using amrex::Real;

    const Real half = Real(0.5);
    Real uy = half*( vy(i,j,k,0) + vy(i,j-1,k,0) );
    Real epsy = dt*dyinv*uy;
    Real nuyy = nudiff + (Real(1.0/3.0)*epsy*epsy); 
    for(int n = URHO; n < conscomp; ++n){
        fldy(i,j,k,n) = nuyy*( u0(i,j,k,n) - u0(i,j-1,k,n) );
    }
}

AMREX_GPU_DEVICE
inline
void
fct_ad_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& vx,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real const> const& u0,
                amrex::Array4<amrex::Real> const& ucx,
                amrex::Real dxinv, amrex::Real dt, 
                amrex::Real diff1, const int conscomp,
                const amrex::Real mudiff) noexcept
{
    using amrex::Real;

    Real ux      = Real(0.5)*( vx(i-1,j,k,0) + vx(i,j,k,0) );
    Real epsx    = dt*dxinv*ux;
    Real muxx    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsx*epsx)); 
    // Real epsy = dt*dxinv*Real(0.125)
    //           *( (ubord(i-1,j-1,k,UMY)/ubord(i-1,j-1,k,URHO)) 
    //           + (Real(2.0)*ubord(i-1,j,k,UMY)/ubord(i-1,j,k,URHO)) 
    //           + (ubord(i-1,j+1,k,UMY)/ubord(i-1,j+1,k,URHO))
    //           + (ubord(i,j-1,k,UMY)/ubord(i,j-1,k,URHO)) 
    //           + (Real(2.0)*ubord(i,j,k,UMY)/ubord(i,j,k,URHO))
    //           + (ubord(i,j+1,k,UMY)/ubord(i,j+1,k,URHO)) );
    // Real muxy = -Real(0.5)*epsx*epsy*cross_ad;
    for(int n = URHO; n < conscomp; ++n){
        Real resdiff  = mudiff * (u0(i,j,k,n) - u0(i-1,j,k,n));
        flax(i,j,k,n) = diff1*( (muxx*(ucx(i,j,k,n) -  ucx(i-1,j,k,n))) ) - resdiff ;
                      // + Real(0.25)*muxy*( ucx(i-1,j,k,n)+ucx(i,j,k,n)+ucx(i-1,j+1,k,n)+ucx(i,j+1,k,n)
                      // - ucx(i-1,j-1,k,n)- ucx(i,j-1,k,n)-ucx(i-1,j,k,n)-ucx(i,j,k,n) ) );
    }
}

AMREX_GPU_DEVICE
inline
void
fct_ad_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& vy,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real const> const& u0,
                amrex::Array4<amrex::Real> const& ucy,
                amrex::Real dyinv, amrex::Real dt,
                amrex::Real diff1, const int conscomp,
                const amrex::Real mudiff) noexcept
{
    using amrex::Real;

    Real uy      = Real(0.5)*( vy(i,j-1,k,0) + vy(i,j,k,0) );
    Real epsy    = dt*dyinv*uy;
    Real muyy    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsy*epsy));
    // Real epsx = dt*dxinv*Real(0.125)
    //           *( (ubord(i-1,j-1,k,UMX)/ubord(i-1,j-1,k,URHO)) 
    //           + (Real(2.0)*ubord(i,j-1,k,UMX)/ubord(i,j-1,k,URHO)) 
    //           + (ubord(i+1,j-1,k,UMX)/ubord(i+1,j-1,k,URHO))
    //           + (ubord(i-1,j,k,UMY)/ubord(i-1,j,k,URHO)) 
    //           + (Real(2.0)*ubord(i,j,k,UMY)/ubord(i,j,k,URHO))
    //           + (ubord(i+1,j,k,UMY)/ubord(i+1,j,k,URHO)) );
    // Real muxy = -Real(0.5)*epsx*epsy*cross_ad; 
    for(int n = URHO; n < conscomp; ++n){
        Real resdiff = mudiff * (u0(i,j,k,n) - u0(i,j-1,k,n));
        flay(i,j,k,n) = diff1*(muyy*(ucy(i,j,k,n) -  ucy(i,j-1,k,n))) - resdiff ;
                      // + Real(0.25)*muxy*( ucy(i,j-1,k,n)+ucy(i+1,j-1,k,n)+ucy(i+1,j,k,n)+ucy(i,j,k,n)
                      // - ucy(i,j-1,k,n)-ucy(i-1,j-1,k,n)-ucy(i-1,j,k,n)-ucy(i,j,k,n) ));
    }
}

AMREX_GPU_DEVICE
inline
void
fct_prelimit_ad_flux_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& udx) noexcept
{
    using amrex::Real;

    Real fltmp    = amrex::Math::abs(flax(i,j,k,n));
    Real sgn      = amrex::Math::copysign(1.0, udx(i,j,k,n) - udx(i-1,j,k,n));
    flax(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udx(i-1,j,k,n) - udx(i-2,j,k,n)),
                                   sgn*(udx(i+1,j,k,n) - udx(i,j,k,n))  ));     
}

AMREX_GPU_DEVICE
inline
void
fct_prelimit_ad_flux_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& udy) noexcept
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
fct_compute_frac_fluxes (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                amrex::Array4<amrex::Real> const& ud) noexcept
{
    using amrex::Real;

    Real fintmp  = amrex::max(flax(i,j,k,n),Real(0.0)) - amrex::min(flax(i+1,j,k,n),Real(0.0))
                 + amrex::max(flay(i,j,k,n),Real(0.0)) - amrex::min(flay(i,j+1,k,n),Real(0.0));

    Real fouttmp = amrex::max(flax(i+1,j,k,n),Real(0.0)) - amrex::min(flax(i,j,k,n),Real(0.0))
                 + amrex::max(flay(i,j+1,k,n),Real(0.0)) - amrex::min(flay(i,j,k,n),Real(0.0));

    Real umin    = amrex::min(ud(i-1,j,k,n),ud(i,j-1,k,n),ud(i,j,k,n),
                              ud(i+1,j,k,n),ud(i,j+1,k,n));
    Real umax    = amrex::max(ud(i-1,j,k,n),ud(i,j-1,k,n),ud(i,j,k,n),
                              ud(i+1,j,k,n),ud(i,j+1,k,n)); 

    frin(i,j,k,n)  = (umax - ud(i,j,k,n))/(Real(1E-50) + fintmp);
    frout(i,j,k,n) = (ud(i,j,k,n) - umin)/(Real(1E-50) + fouttmp); 
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
        flax(i,j,k,n) = fltmp*amrex::min( frin(i-1,j,k,n),frout(i,j,k,n),Real(1.0));    
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
        flay(i,j,k,n) = fltmp*amrex::min( frin(i,j-1,k,n),frout(i,j,k,n),Real(1.0));
    
}

#endif