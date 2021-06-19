#ifndef CNS_TAGGING_H_
#define CNS_TAGGING_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_TagBox.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE
inline
void
cns_get_grad    (int i, int j, int k,
                  amrex::Array4<amrex::Real const> const& S,
                  amrex::Array4<amrex::Real> const& grad,
                  amrex::GeometryData const& geomdata,
                  int wc) noexcept
{
    const amrex::Real* dx      = geomdata.CellSize();
    amrex::Real ax = amrex::Math::abs(S(i+1,j,k,wc) - S(i,j,k,wc));
    amrex::Real ay = amrex::Math::abs(S(i,j+1,k,wc) - S(i,j,k,wc));
    grad(i,j,k,0) = amrex::max(ax,amrex::Math::abs(S(i,j,k,wc) - S(i-1,j,k,wc)))/dx[0];
    grad(i,j,k,1) = amrex::max(ay,amrex::Math::abs(S(i,j,k,wc) - S(i,j-1,k,wc)))/dx[1];
#if AMREX_SPACEDIM==2
        grad(i,j,k,2) = sqrt(grad(i,j,k,0)*grad(i,j,k,0) + grad(i,j,k,1)*grad(i,j,k,1));
#elif AMREX_SPACEDIM==3
        amrex::Real az = amrex::Math::abs(S(i,j,k+1,wc) - S(i,j,k,wc));
        grad(i,j,k,2) = amrex::max(az,amrex::Math::abs(S(i,j,k,wc) - S(i,j,k-1,wc)))/dx[2];
        grad(i,j,k,3) = sqrt(grad(i,j,k,0)*grad(i,j,k,0) + grad(i,j,k,1)*grad(i,j,k,1) + 
                             grad(i,j,k,2)*grad(i,j,k,2));
#endif 
}

AMREX_GPU_HOST_DEVICE
inline
void
cns_tag_maxgrad   (int i, int j, int k,
                  amrex::Array4<char> const& tag,
                  amrex::Array4<amrex::Real> const& grad,
                  amrex::Real grad_threshold, char tagval,
                  amrex::Real tagfrac) noexcept
{
    amrex::Real maxgrad = 0.0;
    amrex::Real gradx = amrex::Math::abs(grad(i,j,k,0));
    amrex::Real grady = amrex::Math::abs(grad(i,j,k,1));
#if AMREX_SPACEDIM==2    
    maxgrad = amrex::max(gradx, grady);
#elif AMREX_SPACEDIM==3
        amrex::Real gradz = amrex::Math::abs(grad(i,j,k,2));
        maxgrad = amrex::max(gradx, grady, gradz);
#endif
    if(maxgrad >= tagfrac*grad_threshold) {
        tag(i,j,k) = tagval;
    } 
}

AMREX_GPU_HOST_DEVICE
inline
void
cns_tag_graderror (int i, int j, int k,
                  amrex::Array4<char> const& tag,
                  amrex::Array4<amrex::Real const> const& S,
                  amrex::GeometryData const& geomdata,
                  amrex::Real threshold_val, char tagval, int wc) noexcept
{
    const amrex::Real* dx      = geomdata.CellSize();
    amrex::Real ax = amrex::Math::abs(S(i+1,j,k,wc) - S(i,j,k,wc));
    amrex::Real ay = amrex::Math::abs(S(i,j+1,k,wc) - S(i,j,k,wc));
    amrex::Real az = 0.0;
    ax = amrex::max(ax,amrex::Math::abs(S(i,j,k,wc) - S(i-1,j,k,wc)))/dx[0];
    ay = amrex::max(ay,amrex::Math::abs(S(i,j,k,wc) - S(i,j-1,k,wc)))/dx[1];
    amrex::Real maxgrad = 0.0;

#if AMREX_SPACEDIM==2    
    maxgrad = amrex::max(ax, ay);
#elif AMREX_SPACEDIM==3
    az = amrex::Math::abs(S(i,j,k+1,wc) - S(i,j,k,wc));
    az = amrex::max(az,amrex::Math::abs(S(i,j,k,wc) - S(i,j,k-1,wc)))/dx[2];
    maxgrad = amrex::max(ax, ay, az);
#endif
    
    if (maxgrad >= threshold_val) {
        tag(i,j,k) = tagval;
    }
}

#endif
