
#include "CNS.H"
#if AMREX_SPACEDIM==2
#include "CNS_hydro_2D_K.H"
#ifdef LRFPFCT_REACTION
#include "LRFPFCT_reaction_2D_K.H"
#endif
#include "LRFPFCT_diffusion_2D_K.H"
#else
#include "CNS_hydro_3D_K.H"
#ifdef LRFPFCT_REACTION
#include "LRFPFCT_reaction_3D_K.H"
#endif
#include "LRFPFCT_diffusion_3D_K.H"
#endif
#include "CNS_K.H"

void 
CNS::compute_dSdt(const MultiFab& Sold, MultiFab& S, MultiFab& dSdt, Real dt, Real dtvel,
                  AMREX_D_DECL(MultiFab& flux, MultiFab& fluy, MultiFab& fluz),
                  FluxRegister* fr_as_crse, FluxRegister* fr_as_fine, int rk)
{
    BL_PROFILE("CNS::compute_dSdt");

    const Box& domain = geom.Domain();
    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();

    const Real diff = diff1;

    Real dtinv = Real(1.0)/dt;
    Real dtvinv = Real(1.0)/dtvel;

    EOSParm const* leosparm = d_eos_parm;
#ifdef LRFPFCT_REACTION
    Real dtsub = dt / ((Real) react_nsubcycle);
#endif

    FArrayBox flt[BL_SPACEDIM], fld[BL_SPACEDIM], 
              ut[BL_SPACEDIM], ud[BL_SPACEDIM], 
              vel[BL_SPACEDIM], utmp, usrc, fracin, fracou;
#ifdef LRFPFCT_REACTION
    FArrayBox omegarho;
#endif
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();

        auto const& sofab = Sold.array(mfi);
        auto const& sfab  = S.array(mfi);
        auto const& dsdtfab = dSdt.array(mfi);

        int ngtmp = NUM_GROW-1;
        const int react_do = do_react;
        const int res_diff = do_fct_resdiff;
        int conscomp = CCOMP;

        const Box& bxg = amrex::grow(bx,ngtmp);

        utmp.resize(bxg,conscomp);
        Elixir utmpeli = utmp.elixir();
        auto const& udfab = utmp.array();

        Elixir flteli[BL_SPACEDIM], fldeli[BL_SPACEDIM], veleli[BL_SPACEDIM], uteli[BL_SPACEDIM],
               udeli[BL_SPACEDIM];
        for (int dir = 0; dir < AMREX_SPACEDIM ; dir++) {
            const Box& bxtmp = amrex::surroundingNodes(bx,dir);
            flt[dir].resize(amrex::grow(bxtmp,3),conscomp);  
            flteli[dir] = flt[dir].elixir();

            fld[dir].resize(amrex::grow(bxtmp,3),conscomp);  
            fldeli[dir] = fld[dir].elixir();

            vel[dir].resize(amrex::grow(bxg,1),1);  
            veleli[dir] = vel[dir].elixir();

            ut[dir].resize(bxg,conscomp);  
            uteli[dir] = ut[dir].elixir();

            ud[dir].resize(bxg,conscomp);  
            udeli[dir] = ud[dir].elixir();
        }

        GpuArray<Array4<Real>, AMREX_SPACEDIM> vfab{ AMREX_D_DECL(vel[0].array(), 
                                                vel[1].array(), vel[2].array())};

        AMREX_D_TERM(auto const& fxfab = flux.array(mfi);,
                     auto const& fyfab = fluy.array(mfi);,
                     auto const& fzfab = fluz.array(mfi););

        GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL(flt[0].array(), 
                                                flt[1].array(), flt[2].array())}; 
        GpuArray<Array4<Real>, AMREX_SPACEDIM> fldx{ AMREX_D_DECL(fld[0].array(), 
                                                fld[1].array(), fld[2].array())}; 
        GpuArray<Array4<Real>, AMREX_SPACEDIM> utr{ AMREX_D_DECL(ut[0].array(), 
                                                ut[1].array(), ut[2].array())}; 
        GpuArray<Array4<Real>, AMREX_SPACEDIM> udi{ AMREX_D_DECL(ud[0].array(), 
                                                ud[1].array(), ud[2].array())}; 

        // Compute the cell-centred velocities
        amrex::ParallelFor(amrex::grow(bx,NUM_GROW),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   Real rhoinv = 1.0/sfab(i,j,k,URHO);
            AMREX_D_TERM(vfab[0](i,j,k,0) = rhoinv*sfab(i,j,k,UMX);,
                         vfab[1](i,j,k,0) = rhoinv*sfab(i,j,k,UMY);,
                         vfab[2](i,j,k,0) = rhoinv*sfab(i,j,k,UMZ););  });

        // --------------Computing the convective fluxes-----------------------
        // compute the x-convective fluxes
        const Box& bxx = amrex::grow(amrex::surroundingNodes(bx,0),NUM_GROW-1);
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_x(i, j, k, vfab[0], fltx[0], sfab);   });

        // compute the y-convective fluxes
        const Box& bxy = amrex::grow(amrex::surroundingNodes(bx,1),NUM_GROW-1);
        amrex::ParallelFor(bxy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_y(i, j, k, vfab[1], fltx[1], sfab);   });

#if AMREX_SPACEDIM==3
        //  Compute z-convective fluxes here
        const Box& bxz = amrex::grow(amrex::surroundingNodes(bx,2),NUM_GROW-1);
        amrex::ParallelFor(bxz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_z(i, j, k, vfab[2], fltx[2], sfab);   });
#endif
        // --------------Computing the diffusion fluxes--------------------
        Real nudiff = 0.0;
        if(res_diff==1) nudiff = 1.0/12.0;
        else            nudiff = 1.0/6.0;
        // compute the x-diffusion fluxes
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_x(i, j, k, vfab[0], fldx[0], sfab, dxinv[0], dt, conscomp, nudiff);   });
        // compute the y-diffusion fluxes
        amrex::ParallelFor(bxy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_y(i, j, k, vfab[1], fldx[1], sfab, dxinv[1], dt, conscomp, nudiff);   });

#if AMREX_SPACEDIM==3
        // Compute z-diffusion fluxes here
        amrex::ParallelFor(bxz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_z(i, j, k, vfab[2], fldx[2], sfab, dxinv[2], dt, conscomp, nudiff);   });
#endif

#ifdef LRFPFCT_REACTION
        omegarho.resize(bxg,1);  
        Elixir omegeli = omegarho.elixir();
        auto const& omgfab = omegarho.array();

        const int nsub = react_nsubcycle;
        for(int ns = 0; ns < nsub; ++ns){
            // First, calculate the reaction rates
            amrex::ParallelFor(bxg,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                fct_reaction_terms(i, j, k, omgfab, sfab, dtsub, nsub, *leosparm); 
            }); 
        }

#endif
        // Obtain the low-order solution, transported quantities, diffused quantitites
        // obtain the contribution of low-order solution to the RHS (dSdt)
        amrex::ParallelFor(bxg, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            AMREX_D_TERM(
            utr[0](i,j,k,n) = sofab(i,j,k,n) - dtvel*dxinv[0]*(fltx[0](i+1,j,k,n)-fltx[0](i,j,k,n));,
            utr[1](i,j,k,n) = sofab(i,j,k,n) - dtvel*dxinv[1]*(fltx[1](i,j+1,k,n)-fltx[1](i,j,k,n));,
            utr[2](i,j,k,n) = sofab(i,j,k,n) - dtvel*dxinv[2]*(fltx[2](i,j,k+1,n)-fltx[2](i,j,k,n));
            );

            AMREX_D_TERM(
            udi[0](i,j,k,n) = utr[0](i,j,k,n) + (fldx[0](i+1,j,k,n)-fldx[0](i,j,k,n));,
            udi[1](i,j,k,n) = utr[1](i,j,k,n) + (fldx[1](i,j+1,k,n)-fldx[1](i,j,k,n));,
            udi[2](i,j,k,n) = utr[2](i,j,k,n) + (fldx[2](i,j,k+1,n)-fldx[2](i,j,k,n));
            );

            udfab(i,j,k,n)  = sofab(i,j,k,n) - dtvel*dxinv[0]*(fltx[0](i+1,j,k,n) - fltx[0](i,j,k,n))
                                             + (fldx[0](i+1,j,k,n) - fldx[0](i,j,k,n))
                                             - dtvel*dxinv[1]*(fltx[1](i,j+1,k,n) - fltx[1](i,j,k,n))
                                             + (fldx[1](i,j+1,k,n) - fldx[1](i,j,k,n))
#if AMREX_SPACEDIM==3
                                             - dtvel*dxinv[2]*(fltx[2](i,j,k+1,n) - fltx[2](i,j,k,n))
                                             + (fldx[2](i,j,k+1,n) - fldx[2](i,j,k,n))
#endif
                                             ;

            dsdtfab(i,j,k,n) = -dt*dxinv[0]*(fltx[0](i+1,j,k,n) - fltx[0](i,j,k,n)) 
                             +     (fldx[0](i+1,j,k,n) - fldx[0](i,j,k,n))
                             -  dt*dxinv[1]*(fltx[1](i,j+1,k,n) - fltx[1](i,j,k,n))
                             +     (fldx[1](i,j+1,k,n) - fldx[1](i,j,k,n))
#if AMREX_SPACEDIM==3
                             -  dt*dxinv[2]*(fltx[2](i,j,k+1,n) - fltx[2](i,j,k,n))
                             +     (fldx[2](i,j,k+1,n) - fldx[2](i,j,k,n))
#endif
                             ;

        }); 

#ifdef LRFPFCT_REACTION
        // Add the reaction source terms to the low-order solution
        amrex::ParallelFor(bxg,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            fct_add_reaction_source_terms(i, j, k, AMREX_D_DECL(udi[0], udi[1], udi[2]), dsdtfab,
                                          omgfab, udfab, *leosparm, dtvel);
        }); 

#endif

    // Store the low-order fluxes in the fluxes MultiFab (needed for refluxing)
        if(rk >= 1){
            const Box& bxd = amrex::surroundingNodes(bx,0);
            amrex::ParallelFor(bxd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fxfab(i,j,k,n) = fltx[0](i,j,k,n) - dx[0]*dtinv*fldx[0](i,j,k,n); });

            const Box& byd = amrex::surroundingNodes(bx,1);
            amrex::ParallelFor(byd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fyfab(i,j,k,n) = fltx[1](i,j,k,n) - dx[1]*dtinv*fldx[1](i,j,k,n); });

#if AMREX_SPACEDIM==3
            const Box& bzd = amrex::surroundingNodes(bx,2);
            amrex::ParallelFor(bzd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fzfab(i,j,k,n) = fltx[2](i,j,k,n) - dx[2]*dtinv*fldx[2](i,j,k,n); });
#endif
        }

        // Compute the anti-diffusive fluxes and store these in fltx
        // --------------Computing the anti-diffusion fluxes--------------------
        Real mudiff = 0.0;
        if(res_diff==1) mudiff = 1.0/12.0;
        // compute the x-diffusion fluxes
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_x(i, j, k, vfab[0], fltx[0], sfab, utr[0], dxinv[0], dt, diff, conscomp, mudiff);   });
        // compute the y-diffusion fluxes
        amrex::ParallelFor(bxy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_y(i, j, k, vfab[1], fltx[1], sfab, utr[1], dxinv[1], dt, diff, conscomp, mudiff);   });

#if AMREX_SPACEDIM==3
        // add the part to compute z-antidiffusion fluxes here
#endif

        // Prelimit anti-diffusion fluxes
        const Box& bxnd1 = amrex::grow(amrex::surroundingNodes(bx,0),1);
        amrex::ParallelFor(bxnd1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_x(i, j, k, n, fltx[0], udi[0]); });

        const Box& bynd1 = amrex::grow(amrex::surroundingNodes(bx,1),1);
        amrex::ParallelFor(bynd1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_y(i, j, k, n, fltx[1], udi[1]); });

        // compute the z-antidiffusive fluxes
#if AMREX_SPACEDIM==3
        const Box& bznd2 = amrex::grow(amrex::surroundingNodes(bx,2),2);         
        amrex::ParallelFor(bznd2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_z(i, j, k, fltx[2], sofab, snfab, sczfab, dxinv[2], dt, diff, conscomp, muadiff); });

         const Box& bznd1 = amrex::grow(amrex::surroundingNodes(bx,2),1);
        amrex::ParallelFor(bznd1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_z(i, j, k, n, fltx[2], udi[2]); });            
#endif

        // Calculate the total incoming and outgoing antidiffusive fluxes in each cell
        fracin.resize(amrex::grow(bx,1),conscomp);
        Elixir fineli = fracin.elixir();
        auto const& finfab = fracin.array();

        fracou.resize(amrex::grow(bx,1),conscomp);
        Elixir foueli = fracou.elixir();
        auto const& foufab = fracou.array();

        const Box& bxg1 = amrex::grow(bx,1);
        amrex::ParallelFor(bxg1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_compute_frac_fluxes(i, j, k, n, AMREX_D_DECL(fltx[0], fltx[1], fltx[2]), 
                                        finfab, foufab, udfab);  });

         // --------------- Compute the corrected fluxes (no ghost cells) -------------------- 
        const Box& bxnd = amrex::surroundingNodes(bx,0);
        amrex::ParallelFor(bxnd, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_correct_fluxes_x(i, j, k, n, fltx[0], finfab, foufab); });

        const Box& bynd = amrex::surroundingNodes(bx,1);
        amrex::ParallelFor(bynd, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_correct_fluxes_y(i, j, k, n, fltx[1], finfab, foufab); });

        // --------------- Compute the corrected z-fluxes (no ghost cells) -------------------- 
#if AMREX_SPACEDIM==3
        const Box& bznd = amrex::surroundingNodes(bx,2);
        amrex::ParallelFor(bznd, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_correct_fluxes_z(i, j, k, n, fltx[2], finfab, foufab); });                
#endif

        // Compute the RHS (dsdtfab)
        amrex::ParallelFor(bx, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   dsdtfab(i,j,k,n) = dsdtfab(i,j,k,n)
                             + fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n) 
                             + fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n)
#if AMREX_SPACEDIM==3
                             + fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n)
#endif
                                                                        ;
        }); 

        // Store the fluxes in fluxes MultiFab
        if(rk>=1){
            const Box& bxd = amrex::surroundingNodes(bx,0);
            amrex::ParallelFor(bxd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fxfab(i,j,k,n) += dx[0]*dtinv*fltx[0](i,j,k,n); });

            const Box& byd = amrex::surroundingNodes(bx,1);
            amrex::ParallelFor(byd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fyfab(i,j,k,n) += dx[1]*dtinv*fltx[1](i,j,k,n); });

#if AMREX_SPACEDIM==3
            const Box& bzd = amrex::surroundingNodes(bx,2);
            amrex::ParallelFor(bzd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fzfab(i,j,k,n) += dx[2]*dtinv*fltx[2](i,j,k,n); });
#endif

        // scale the fluxes now
            amrex::ParallelFor(bxd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   
#if AMREX_SPACEDIM==2
                fxfab(i,j,k,n) = fxfab(i,j,k,n)*dx[1]*dt;
#else
                fxfab(i,j,k,n) = fxfab(i,j,k,n)*dx[1]*dx[2]*dt;
#endif
            });

            amrex::ParallelFor(byd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   
#if AMREX_SPACEDIM==2
                fyfab(i,j,k,n) = fyfab(i,j,k,n)*dx[0]*dt;
#else
                fyfab(i,j,k,n) = fyfab(i,j,k,n)*dx[0]*dx[2]*dt;
#endif
            });

#if AMREX_SPACEDIM==3
            amrex::ParallelFor(bzd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fzfab(i,j,k,n) = fzfab(i,j,k,n)*dx[0]*dx[1]*dt; });
#endif  
        }

    }

    // This is needed to initialize the flux registers for refluxing (only when AMR is used)
    if(rk >= 1){
        int conscomp = CCOMP;
        if (fr_as_crse) {
                AMREX_D_TERM(fr_as_crse->CrseInit(flux, 0, 0, 0, conscomp, Real(-1.0), FluxRegister::ADD);,
                             fr_as_crse->CrseInit(fluy, 1, 0, 0, conscomp, Real(-1.0), FluxRegister::ADD);,
                             fr_as_crse->CrseInit(fluz, 2, 0, 0, conscomp, Real(-1.0), FluxRegister::ADD););
        }

        if (fr_as_fine) {
                AMREX_D_TERM(fr_as_fine->FineAdd(flux, 0, 0, 0, conscomp, Real(1.0));,
                             fr_as_fine->FineAdd(fluy, 1, 0, 0, conscomp, Real(1.0));,
                             fr_as_fine->FineAdd(fluz, 2, 0, 0, conscomp, Real(1.0)););
        }        
    } 
}