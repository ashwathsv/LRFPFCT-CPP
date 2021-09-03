
#include "CNS.H"
#if AMREX_SPACEDIM==2
#include "CNS_hydro_2D_K.H"
#include "LRFPFCT_reaction_2D_K.H"
#else
#include "CNS_hydro_3D_K.H"
#include "LRFPFCT_reaction_3D_K.H"
#endif
#include "CNS_K.H"

using namespace amrex;

Real
CNS::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("CNS::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);

    MultiFab Sborder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());
    MultiFab Soldtmp(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());

    int conscomp = CCOMP;
    if(do_react == 1) conscomp = conscomp + 1; 

    AMREX_D_TERM(MultiFab flux;, MultiFab fluy;, MultiFab fluz;);
    AMREX_D_TERM(flux.define(amrex::convert(S_new.boxArray(),IntVect::TheDimensionVector(0)),
                            S_new.DistributionMap(), conscomp, 0);,
                 fluy.define(amrex::convert(S_new.boxArray(),IntVect::TheDimensionVector(1)),
                            S_new.DistributionMap(), conscomp, 0);,
                 fluz.define(amrex::convert(S_new.boxArray(),IntVect::TheDimensionVector(2)),
                            S_new.DistributionMap(), conscomp, 0););
    AMREX_D_TERM(flux.setVal(Real(0.0));, fluy.setVal(0.0);, fluz.setVal(0.0););

    AMREX_D_TERM(MultiFab Scx;, MultiFab Scy;, MultiFab Scz;);
    AMREX_D_TERM(Scx.define(S_new.boxArray(), S_new.DistributionMap(), conscomp, 3);,
                 Scy.define(S_new.boxArray(), S_new.DistributionMap(), conscomp, 3);,
                 Scz.define(S_new.boxArray(), S_new.DistributionMap(), conscomp, 3););

    AMREX_D_TERM(MultiFab Sdx;, MultiFab Sdy;, MultiFab Sdz;);
    AMREX_D_TERM(Sdx.define(S_new.boxArray(), S_new.DistributionMap(), conscomp, 3);,
                 Sdy.define(S_new.boxArray(), S_new.DistributionMap(), conscomp, 3);,
                 Sdz.define(S_new.boxArray(), S_new.DistributionMap(), conscomp, 3););

    AMREX_D_TERM(Scx.setVal(Real(0.0));, Scy.setVal(0.0);, Scz.setVal(0.0););
    AMREX_D_TERM(Sdx.setVal(Real(0.0));, Sdy.setVal(0.0);, Sdz.setVal(0.0););

    FluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        CNS& fine_level = getLevel(level+1);
        fr_as_crse = fine_level.flux_reg.get();
    }

    FluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = flux_reg.get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(Real(0.0));
    }

    // RK2 stage 1
    int rk = 1;
    FillPatch(*this,Sborder,NUM_GROW,time,State_Type,0,NUM_STATE);
    MultiFab::Copy(Soldtmp,Sborder,0,0,NUM_STATE,NUM_GROW);

    if(Sborder.contains_nan() || Sborder.min(URHO,NUM_GROW) < Real(0.0) 
        || Sborder.min(UEDEN,NUM_GROW) < Real(0.0) || Sborder.min(UPRE,NUM_GROW) < Real(0.0)){
        Print() << "CNS::advance, after FillPatch(), lev = " << level << ", contains NaN() " << "\n";
        Print() << "min(ueden) = " << Sborder.min(UEDEN,NUM_GROW) << ", min(pre) = " << Sborder.min(UPRE,NUM_GROW) << "\n";
        amrex::Error("NaN value found before RK1 advance, aborting...");
    }

    if(do_react == 1 && Sborder.min(URHOY) < Real(0.0)){
        Print() << "CNS::advance, after FillPatch(), lev = " << level << ", negative mass fraction " << "\n";
        Print() << "min(URHOY) = " << Sborder.min(URHOY,NUM_GROW) << "\n";
        amrex::Error("Negative mass fraction found before RK1 advance, aborting...");
    }

    // RK step 1, FCT step 1
    FCT_low_order_solution(Soldtmp, Sborder, S_new, AMREX_D_DECL(Scx, Scy, Scz), 
                 AMREX_D_DECL(Sdx, Sdy, Sdz), AMREX_D_DECL(flux, fluy, fluz),
                 Real(0.5)*dt, fr_as_crse, fr_as_fine, rk);
    // Sborder has the low order solution (solution for upto 3 ghost cells)

    // RK step 1, FCT step 2
    FCT_corrected_solution(Soldtmp, Sborder, S_new, AMREX_D_DECL(Scx, Scy, Scz), 
                 AMREX_D_DECL(Sdx, Sdy, Sdz), AMREX_D_DECL(flux, fluy, fluz),
                 Real(0.5)*dt, fr_as_crse, fr_as_fine, rk);
    // Sborder has the corrected solution for RK1 (no ghost cells)
    // Copy this into S_new MultiFab
    computeTemp(Sborder,0);

    MultiFab::Copy(S_new,Sborder,0,0,NUM_STATE,0);
    computeTemp(S_new,0);

    // add reaction source terms (energy and mass fraction)
    if(do_react == 1){
        reaction_terms_subcycle(S_new, Sborder, 0.5*dt);
    }

    // RK2, FCT step 1
    rk = 2;
    FillPatch(*this,Sborder,NUM_GROW,time+dt,State_Type,0,NUM_STATE);
    FCT_low_order_solution(Soldtmp, Sborder, S_new, AMREX_D_DECL(Scx, Scy, Scz), 
                 AMREX_D_DECL(Sdx, Sdy, Sdz), AMREX_D_DECL(flux, fluy, fluz),
                 dt, fr_as_crse, fr_as_fine, rk);

    // Sborder has the low order solution (solution for upto 3 ghost cells)

    // RK step 2, FCT step 2
    // The corrected solution is stored in S_new
    FCT_corrected_solution(Soldtmp, Sborder, S_new, AMREX_D_DECL(Scx, Scy, Scz), 
                 AMREX_D_DECL(Sdx, Sdy, Sdz), AMREX_D_DECL(flux, fluy, fluz),
                 dt, fr_as_crse, fr_as_fine, rk);
    computeTemp(S_new,0);
    FillPatch(*this,S_new,NUM_GROW,time+dt,State_Type,0,NUM_STATE);

    return dt;
}

void
CNS::FCT_low_order_solution (const MultiFab& S_old, MultiFab& S, MultiFab& S_new, 
                   AMREX_D_DECL(MultiFab& Scx, MultiFab& Scy, MultiFab& Scz),
                   AMREX_D_DECL(MultiFab& Sdx, MultiFab& Sdy, MultiFab& Sdz),
                   AMREX_D_DECL(MultiFab& flux, MultiFab& fluy, MultiFab& fluz),
                   Real dt, FluxRegister* fr_as_crse, FluxRegister* fr_as_fine, int rk)
{
    BL_PROFILE("CNS::FCT_low_order_solution()");

    const Box& domain = geom.Domain();
    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = NUM_STATE;

    Real dtinv = Real(1.0)/dt;

    EOSParm const* leosparm = d_eos_parm;

    FArrayBox fltmp[BL_SPACEDIM], utmp, usrc;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();

        auto const& sofab = S_old.array(mfi);
        auto const& sfab  = S.array(mfi);
        auto const& snfab = S_new.array(mfi);

        AMREX_D_TERM(auto const& scxfab = Scx.array(mfi);,
                     auto const& scyfab = Scy.array(mfi);,
                     auto const& sczfab = Scz.array(mfi););

        AMREX_D_TERM(auto const& sdxfab = Sdx.array(mfi);,
                     auto const& sdyfab = Sdy.array(mfi);,
                     auto const& sdzfab = Sdz.array(mfi););

        int ngtmp = 3;
        if(level > 0) ngtmp = NUM_GROW-1;

        const int react_do = do_react;

        int conscomp = CCOMP;
        if(react_do == 1) conscomp = conscomp + 1; 

        utmp.resize(amrex::grow(bx,ngtmp),conscomp);
        Elixir utmpeli = utmp.elixir();
        auto const& utfab = utmp.array(); 
        
        Elixir fltmpeli[BL_SPACEDIM];
        for (int dir = 0; dir < AMREX_SPACEDIM ; dir++) {
            const Box& bxtmp = amrex::surroundingNodes(bx,dir);
            fltmp[dir].resize(amrex::grow(bxtmp,3),conscomp);  
            fltmpeli[dir] = fltmp[dir].elixir();
        }

        AMREX_D_TERM(auto const& fxfab = flux.array(mfi);,
                     auto const& fyfab = fluy.array(mfi);,
                     auto const& fzfab = fluz.array(mfi););

        GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL(fltmp[0].array(), 
                                                fltmp[1].array(), fltmp[2].array())}; 

        // ---------------------- Computing the convective fluxes ----------------------------------
        // compute the x-convective fluxes 
        const Box& bxx = amrex::grow(amrex::surroundingNodes(bx,0),3);
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_x(i, j, k, fltx[0], sofab, sfab);   });
        // update the x-convected quantities
        amrex::ParallelFor(amrex::grow(bx,3), CCOMP,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   scxfab(i,j,k,n) = sofab(i,j,k,n) + dt*dxinv[0]*(fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n));  });

        if(react_do == 1){
            amrex::ParallelFor(bxx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {   fct_react_convflux_x(i, j, k, fltx[0], sofab, sfab);   });
            // update the x-convected quantities
            amrex::ParallelFor(amrex::grow(bx,3),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {   scxfab(i,j,k,URHOY) = sofab(i,j,k,URHOY) + dt*dxinv[0]*(fltx[0](i,j,k,URHOY) - fltx[0](i+1,j,k,URHOY));  });
        }

        // compute the y-convective fluxes
        // IntVect ngivy(AMREX_D_DECL(1,NUM_GROW,1));
        const Box& bxy = amrex::grow(amrex::surroundingNodes(bx,1),3);
        amrex::ParallelFor(bxy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_y(i, j, k, fltx[1], sofab, sfab);   });
        // update the y-convected quantities
        amrex::ParallelFor(amrex::grow(bx,3), CCOMP,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   scyfab(i,j,k,n) = sofab(i,j,k,n) + dt*dxinv[1]*(fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n));  });

        if(react_do == 1){
            amrex::ParallelFor(bxy,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {   fct_react_convflux_y(i, j, k, fltx[1], sofab, sfab);   });
            // update the y-convected quantities
            amrex::ParallelFor(amrex::grow(bx,3),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {   scyfab(i,j,k,URHOY) = sofab(i,j,k,URHOY) + dt*dxinv[1]*(fltx[1](i,j,k,URHOY) - fltx[1](i,j+1,k,URHOY));  });            
        }

#if AMREX_SPACEDIM==3
        // compute the z-convective fluxes
        const Box& bxz = amrex::grow(amrex::surroundingNodes(bx,2),3);        
        amrex::ParallelFor(bxz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {    fct_react_convflux_z(i, j, k, fltx[2], sofab, sfab);   });
        amrex::ParallelFor(amrex::grow(bx,3), CCOMP,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {    sczfab(i,j,k,n) = sofab(i,j,k,n) + dt*dxinv[2]*(fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n));  });   

        if(react_do == 1){
            amrex::ParallelFor(bxz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {    fct_con_flux_z(i, j, k, fltx[2], sofab, sfab);   });
            amrex::ParallelFor(amrex::grow(bx,3),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {    sczfab(i,j,k,URHOY) = sofab(i,j,k,URHOY) + dt*dxinv[2]*(fltx[2](i,j,k,URHOY) - fltx[2](i,j,k+1,URHOY));  });   
        }         
#endif
        // compute the partially convected quantities
        int ngr = 3; if(level > 0) ngr = 3;
        const Box& bxg3 = amrex::grow(bx,ngr);
        amrex::ParallelFor(bxg3, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            utfab(i,j,k,n)  = sofab(i,j,k,n) + dt*dxinv[0]*(fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n))
                                             + dt*dxinv[1]*(fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n))
#if AMREX_SPACEDIM==3
                                             + dt*dxinv[2]*(fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n))
#endif
                                             ;
        }); 

        // store the convective fluxes in fluxes MultiFab
        if(rk == 2){
            const Box& bxd = amrex::surroundingNodes(bx,0);
            amrex::ParallelFor(bxd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fxfab(i,j,k,n) = fltx[0](i,j,k,n);  });

            const Box& byd = amrex::surroundingNodes(bx,1);
            amrex::ParallelFor(byd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fyfab(i,j,k,n) = fltx[1](i,j,k,n);  });

#if AMREX_SPACEDIM==3
            const Box& bzd = amrex::surroundingNodes(bx,2);
            amrex::ParallelFor(bzd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fzfab(i,j,k,n) = fltx[2](i,j,k,n);  });
#endif
        }

        // ----------------------- Compute the reaction source terms ---------------------------------------------
        if(react_do == 1){
            usrc.resize(amrex::grow(bx,ngtmp),1);
        }
        Elixir usrceli = usrc.elixir();
        auto const& srcfab = usrc.array(); 
        if(react_do == 1){
            amrex::ParallelFor(amrex::grow(bx,ngtmp), 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {   fct_reaction_rates(i, j, k, srcfab, sofab, *leosparm);   });
        }

        // ---------------------- Computing the diffusive fluxes (low order solution) ----------------------------------
        // compute the x-diffusive fluxes 
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_x(i, j, k, fltx[0], sofab, sfab, dxinv[0], dt, conscomp);    });
        // compute the 1-D x-diffused quantities
        amrex::ParallelFor(amrex::grow(bx,3), conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   sdxfab(i,j,k,n) = scxfab(i,j,k,n) + fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n);    });

        // compute the y-diffusive fluxes
        amrex::ParallelFor(bxy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_y(i, j, k, fltx[1], sofab, sfab, dxinv[1], dt, conscomp);    });
        // compute the 1-D y-diffused quantities
        amrex::ParallelFor(amrex::grow(bx,3), conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   sdyfab(i,j,k,n) = scyfab(i,j,k,n) + fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n);   });

#if AMREX_SPACEDIM==3
        // compute the z-diffusive fluxes          
        amrex::ParallelFor(bxz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_z(i, j, k, fltx[2], sofab, sfab, dxinv[2], dt, conscomp);   });     
        // compute the 1-D z-diffused quantities
        amrex::ParallelFor(amrex::grow(bx,3), conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   sdzfab(i,j,k,n) = sczfab(i,j,k,n) + fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n);  });       
#endif
        // compute the partially diffusive (1-D) quantities
        // store low-order solution in Sborder and utmp
        const Box& bxn3 = amrex::grow(bx,ngr);
        amrex::ParallelFor(bxn3, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            utfab(i,j,k,n)  = utfab(i,j,k,n)  + fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n)
                                              + fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n)
#if AMREX_SPACEDIM==3
                                              + fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n)
#endif
                                              ;
            sfab(i,j,k,n)   = utfab(i,j,k,n);
        });

        // add reaction source terms to the low-order solution
        if(react_do == 1){
            amrex::ParallelFor(bxn3,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                fct_add_reaction_source(i, j, k, srcfab, AMREX_D_DECL(sdxfab, sdyfab, sdzfab),
                                        utfab, sfab, sofab, *leosparm);
            });
        }

        // store the diffusive fluxes in fluxes MultiFab
        if(rk == 2){
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
        }

    }

}

void
CNS::FCT_corrected_solution (const MultiFab& S_old, MultiFab& S, MultiFab& S_new, 
                   AMREX_D_DECL(MultiFab& Scx, MultiFab& Scy, MultiFab& Scz),
                   AMREX_D_DECL(MultiFab& Sdx, MultiFab& Sdy, MultiFab& Sdz),
                   AMREX_D_DECL(MultiFab& flux, MultiFab& fluy, MultiFab& fluz),
                   Real dt, FluxRegister* fr_as_crse, FluxRegister* fr_as_fine, int rk)
{
    BL_PROFILE("CNS::FCT_corrected_solution()");

    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    Real dtinv = Real(1.0)/dt;
    const Real diff = diff1;

    EOSParm const* leosparm = d_eos_parm;

    FArrayBox fltmp[BL_SPACEDIM], frin, frout;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();

        auto const& sofab = S_old.array(mfi);
        auto const& sfab  = S.array(mfi);
        auto const& snfab = S_new.array(mfi);

        AMREX_D_TERM(auto const& scxfab = Scx.array(mfi);,
                     auto const& scyfab = Scy.array(mfi);,
                     auto const& sczfab = Scz.array(mfi););

        AMREX_D_TERM(auto const& sdxfab = Sdx.array(mfi);,
                     auto const& sdyfab = Sdy.array(mfi);,
                     auto const& sdzfab = Sdz.array(mfi););

        const int react_do = do_react;

        int conscomp = CCOMP;
        if(react_do == 1) conscomp = conscomp + 1; 
        
        Elixir fltmpeli[BL_SPACEDIM];
        for (int dir = 0; dir < AMREX_SPACEDIM ; dir++) {
            const Box& bxtmp = amrex::surroundingNodes(bx,dir);
            fltmp[dir].resize(amrex::grow(bxtmp,2),conscomp);  
            fltmpeli[dir] = fltmp[dir].elixir();
        }

        AMREX_D_TERM(auto const& fxfab = flux.array(mfi);,
                     auto const& fyfab = fluy.array(mfi);,
                     auto const& fzfab = fluz.array(mfi););

        GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL( fltmp[0].array(), 
                                                fltmp[1].array(), fltmp[2].array()) }; 

        // ----------- Compute and prelimit the antidiffusive fluxes (for upto 2 ghost cells) -----------
        // compute and prelimit the x-antidiffusive fluxes 
        AMREX_D_TERM(int ilo;, int jlo;, int klo;);
        AMREX_D_TERM(int ihi;, int jhi;, int khi;);

        const Box& bxnd2 = amrex::grow(amrex::surroundingNodes(bx,0),2);
        amrex::ParallelFor(bxnd2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_x(i, j, k, fltx[0], sofab, sfab, scxfab, dxinv[0], dt, diff, conscomp);  });

        // compute the y-antidiffusive fluxes
        const Box& bynd2 = amrex::grow(amrex::surroundingNodes(bx,1),2);
        amrex::ParallelFor(bynd2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_y(i, j, k, fltx[1], sofab, sfab, scyfab, dxinv[1], dt, diff, conscomp);  });

        // ------------------Prelimit the anti-diffusive fluxes------------------
        AMREX_D_TERM(ilo= lbound(bx).x-1;, jlo= lbound(bx).y;, klo= lbound(bx).z); 
        AMREX_D_TERM(ihi= ubound(bx).x+2;, jhi= ubound(bx).y;, khi= ubound(bx).z);
        const Box& bxnd1 = amrex::grow(amrex::surroundingNodes(bx,0),1);
        amrex::ParallelFor(bxnd1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_x(i, j, k, n, fltx[0], sdxfab, AMREX_D_DECL(ilo,jlo,klo),
                AMREX_D_DECL(ihi,jhi,khi)); });

        AMREX_D_TERM(ilo= lbound(bx).x;, jlo= lbound(bx).y-1;, klo= lbound(bx).z); 
        AMREX_D_TERM(ihi= ubound(bx).x;, jhi= ubound(bx).y+2;, khi= ubound(bx).z);
        const Box& bynd1 = amrex::grow(amrex::surroundingNodes(bx,1),1);
        amrex::ParallelFor(bynd1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_y(i, j, k, n, fltx[1], sdyfab, AMREX_D_DECL(ilo,jlo,klo),
                AMREX_D_DECL(ihi,jhi,khi)); });

        // compute the z-antidiffusive fluxes
#if AMREX_SPACEDIM==3
        const Box& bznd2 = amrex::grow(amrex::surroundingNodes(bx,2),2);         
        amrex::ParallelFor(bznd2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_z(i, j, k, fltx[2], sofab, sfab, sczfab, dxinv[2], dt, diff, conscomp); });

        // IntVect ngivz(AMREX_D_DECL(0,0,1));
         const Box& bznd1 = amrex::grow(amrex::surroundingNodes(bx,2),1);
        AMREX_D_TERM(ilo= lbound(bx).x;, jlo= lbound(bx).y;, klo= lbound(bx).z-1); 
        AMREX_D_TERM(ihi= ubound(bx).x;, jhi= ubound(bx).y;, khi= ubound(bx).z+2);
        amrex::ParallelFor(bznd1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_z(i, j, k, n, fltx[2], sdzfab, AMREX_D_DECL(ilo,jlo,klo),
                                        AMREX_D_DECL(ihi,jhi,khi)); });            
#endif

        // ----------- Compute the total incoming and outgoing fluxes (for upto 1 ghost cells) -----------
        frin.resize(amrex::grow(bx,1),conscomp); 
        frout.resize(amrex::grow(bx,1),conscomp);  
        Elixir frinelei = frin.elixir();
        Elixir frouteli = frout.elixir();
        auto const& finfab = frin.array();
        auto const& foutfab = frout.array();

        const Box& bxg1 = amrex::grow(bx,1);
        amrex::ParallelFor(bxg1, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_compute_frac_fluxes(i, j, k, n, AMREX_D_DECL(fltx[0], fltx[1], fltx[2]), 
                                        finfab, foutfab, sfab);  });

        // --------------- Compute the corrected fluxes (no ghost cells) -------------------- 
        const Box& bxnd = amrex::surroundingNodes(bx,0);
        amrex::ParallelFor(bxnd, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_correct_fluxes_x(i, j, k, n, fltx[0], finfab, foutfab); });

        const Box& bynd = amrex::surroundingNodes(bx,1);
        amrex::ParallelFor(bynd, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_correct_fluxes_y(i, j, k, n, fltx[1], finfab, foutfab); });

        // Print() << "reached here, completed correcting anti-diffusive flux for RK1  \n";

        // --------------- Compute the corrected z-fluxes (no ghost cells) -------------------- 
#if AMREX_SPACEDIM==3
        const Box& bznd = amrex::surroundingNodes(bx,2);
        amrex::ParallelFor(bznd, conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_correct_fluxes_z(i, j, k, n, fltx[2], finfab, foutfab); });                
#endif

        // --------------- Compute corrected solution and store in Sborder(RK1)
        // ----------------or Snew(RK2) (no ghost cells) --------------------
        if(rk == 1){
            amrex::ParallelFor(bx, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {   sfab(i,j,k,n)    = sfab(i,j,k,n)
                                 + fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n) 
                                 + fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n)
#if AMREX_SPACEDIM==3
                                 + fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n)
#endif
                                                                        ;
            }); 
        }else{  amrex::ParallelFor(bx, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {   snfab(i,j,k,n)   = sfab(i,j,k,n)
                                 + fltx[0](i,j,k,n) - fltx[0](i+1,j,k,n) 
                                 + fltx[1](i,j,k,n) - fltx[1](i,j+1,k,n)
#if AMREX_SPACEDIM==3
                                 + fltx[2](i,j,k,n) - fltx[2](i,j,k+1,n)
#endif
                                                                        ;
            });
        }

        // store the corrected fluxes in fluxes MultiFab
        if(rk == 2){
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

    if(rk == 2){
        int conscomp = CCOMP;
        if(do_react == 1) conscomp = conscomp + 1;

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

void
CNS::reaction_terms_subcycle(MultiFab& S_new, MultiFab& Sbord, Real dt)
{
    BL_PROFILE("CNS::reaction_terms_subcycle()");

    const Box& domain = geom.Domain();
    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = NUM_STATE;

    Real dtsub = dt / ((Real) react_nsubcycle);

    EOSParm const* leosparm = d_eos_parm;

    FArrayBox fltmp[BL_SPACEDIM], utmp, usrc;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();

        auto const& sfab = S_new.array(mfi);
        auto const& sbfab  = Sbord.array(mfi);

        int ngtmp = 3;
        if(level > 0) ngtmp = NUM_GROW-1;

        const int nsub = react_nsubcycle;

        // ----------------------- Compute the reaction source terms ---------------------------------------------

        // add reaction source terms to the low-order solution
        if(react_do == 1){
            amrex::ParallelFor(bxn3,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                fct_add_reaction_source(i, j, k, sfab, sbfab, dtsub, nsub, *leosparm);
            });
        }

    }

}

