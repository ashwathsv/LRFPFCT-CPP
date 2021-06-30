#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"

// #ifdef AMREX_USE_GPU
AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();


    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
#if AMREX_SPACEDIM==3
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif

    // if(prob_parm.geom_type_flag == 1){
        AMREX_D_TERM(Real xcm;, Real ycm;, Real zcm);
        AMREX_D_TERM(xcm = 0.5*(prob_lo[0] + prob_hi[0]);,
                     ycm = 0.5*(prob_lo[1] + prob_hi[1]);,
                     zcm = 0.5*(prob_lo[2] + prob_hi[2]););

        Real radius = 0.5*prob_parm.width;

        Real dist = 0.0;
#if AMREX_SPACEDIM==2
            dist = (x-xcm)*(x-xcm) + (y-ycm)*(y-ycm) - radius*radius;
#else
            dist = (x-xcm)*(x-xcm) + (y-ycm)*(y-ycm) + (z-zcm)*(z-zcm) - radius*radius;
#endif

        Real Pt, rhot, uxt, uyt, uzt;
        if(dist < 0.0){
            Pt = prob_parm.p_l;
            rhot = prob_parm.rho_l;
            uxt = prob_parm.u_l;
            uyt = prob_parm.v_l;
#if AMREX_SPACEDIM==3
            uzt = prob_parm.w_l;
#endif
        }else{
            Pt = prob_parm.p_r;
            rhot = prob_parm.rho_r;
            uxt = prob_parm.u_r;
            uyt = prob_parm.v_r;
#if AMREX_SPACEDIM==3
            uzt = prob_parm.w_r;
#endif        
        }
        state(i,j,k,URHO ) = rhot;
        state(i,j,k,UMX  ) = rhot*uxt;
        state(i,j,k,UMY  ) = rhot*uyt;
#if AMREX_SPACEDIM==3        
        state(i,j,k,UMZ  ) = rhot*uzt;
#endif
        Real et = Pt/(parm.eos_gamma-Real(1.0));
        state(i,j,k,UEINT) = et;
        state(i,j,k,UPRE)  = Pt;

#if AMREX_SPACEDIM==2
            state(i,j,k,UEDEN) = et + Real(0.5)*rhot*(uxt*uxt + uyt*uyt);
#else 
            state(i,j,k,UEDEN) = et + Real(0.5)*rhot*(uxt*uxt + uyt*uyt + uzt*uzt);
#endif
        state(i,j,k,UTEMP) = (Real(1.0)/state(i,j,k,URHO)) * state(i,j,k,UEINT) * (Real(1.0)/parm.cv);
    // }
    
}

AMREX_GPU_DEVICE
inline
void
cns_initdata_pureadv (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();


    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
#if AMREX_SPACEDIM==3
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif

    // if(prob_parm.geom_type_flag == 1){
        AMREX_D_TERM(Real xcm;, Real ycm;, Real zcm);
        AMREX_D_TERM(xcm = 0.5*(prob_lo[0] + prob_hi[0]);,
                     ycm = 0.5*(prob_lo[1] + prob_hi[1]);,
                     zcm = 0.5*(prob_lo[2] + prob_hi[2]););

        Real radius = 0.5*prob_parm.width;

        Real dist = 0.0;
#if AMREX_SPACEDIM==2
            dist = (x-xcm)*(x-xcm) + (y-ycm)*(y-ycm) - radius*radius;
#else
            dist = (x-xcm)*(x-xcm) + (y-ycm)*(y-ycm) + (z-zcm)*(z-zcm) - radius*radius;
#endif

        Real Pt, rhot, uxt, uyt, uzt;
        if(dist < 0.0){
            Pt = Real(0.0);
            rhot = prob_parm.rho_l;
            uxt = prob_parm.u_l;
            uyt = prob_parm.v_l;
#if AMREX_SPACEDIM==3
            uzt = prob_parm.w_l;
#endif
        }else{
            Pt = Real(0.0);
            rhot = prob_parm.rho_r;
            uxt = prob_parm.u_r;
            uyt = prob_parm.v_r;
#if AMREX_SPACEDIM==3
            uzt = prob_parm.w_r;
#endif        
        }
        state(i,j,k,URHO ) = rhot;
        state(i,j,k,UMX  ) = rhot*uxt;
        state(i,j,k,UMY  ) = rhot*uyt;
#if AMREX_SPACEDIM==3        
        state(i,j,k,UMZ  ) = rhot*uzt;
#endif
        Real et = Pt/(parm.eos_gamma-Real(1.0));
        state(i,j,k,UEINT) = et;
        state(i,j,k,UPRE)  = Pt;

#if AMREX_SPACEDIM==2
            state(i,j,k,UEDEN) = et + Real(0.5)*rhot*(uxt*uxt + uyt*uyt);
#else 
            state(i,j,k,UEDEN) = et + Real(0.5)*rhot*(uxt*uxt + uyt*uyt + uzt*uzt);
#endif
        state(i,j,k,UTEMP) = (Real(1.0)/state(i,j,k,URHO)) * state(i,j,k,UEINT) * (Real(1.0)/parm.cv);
    // }
    
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_probspecific_bc(
  const amrex::Real* /*x[AMREX_SPACEDIM]*/,
  const amrex::Real s_int[NGROW][CCOMP],
  amrex::Real s_ext[CCOMP],
  const int /*idir*/,
  const int /*ind*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParm const& /*prob_parm*/)
{

}

#endif
