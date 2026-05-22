/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
*/



// Check development branch, the foundation version reorganized classes after version 6

#if foamVersion < 1000 && foamVersion > 10
    #define newFoundationVersion 1
#else
    #define isFoundationVersion 0
#endif

#if (newFoundationVersion)
   // The content of "fvCFD.H" has been splitted, pick what's necessary below
    #include "argList.H"
    #include "volFields.H"
    #include "fvMesh.H"
    #include "fvSchemes.H"
    #include "fvSolution.H"
    #include "surfaceFields.H"
    #include "fvm.H"

    #include "pressureReference.H"
    #include "findRefCell.H"
    #include "constrainPressure.H"
    #include "constrainHbyA.H"
    #include "adjustPhi.H"

    #include "fvcDdt.H"
    #include "fvcGrad.H"
    #include "fvcFlux.H"

    #include "fvmDdt.H"
    #include "fvmDiv.H"
    #include "fvmLaplacian.H"
    #include "fvcLaplacian.H"
    #include "polyMesh.H"
    #include <fstream>




#else
    #include "fvCFD.H"
#endif

#include "pisoControl.H"
#include "../../FoamYade/FoamYade.H"
#include "lambdaDotModel.H"   //DasteXar jadid


using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #if (newFoundationVersion)
        #include "setRootCase.H"
    #else
        #include "setRootCaseLists.H"
    #endif
    #include "createTime.H"
    #include "createMesh.H"



    pisoControl piso(mesh);

    #include "createFields.H"
    #include "initContinuityErrs.H"


    
    
  
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    bool gaussianInterp = false;

    FoamYade yadeCoupling(mesh,U, gradP, vGrad, divT,ddtU_f,g,uSourceDrag,alphac, uSource, uParticle, uCoeff,uInterp, lambdaDot, gaussianInterp);

    yadeCoupling.setScalarProperties(partDensity.value(), fluidDensity.value(), nu.value());
    std::cout << "done set of part properties" << std::endl;


    lambdaDotModel lambdaDotUpdater(mesh, lambdaDot, nParticles, yadeCoupling); //DasteXar jadid

	yadeCoupling.locateAllParticles();   // places the location of spheres from time 0



    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        #include "CourantNo.H"


        vGrad = fvc::grad(U);
        
       

        yadeCoupling.setParticleAction(runTime.deltaT().value());


        lambdaDotUpdater.update(); //DasteXar jadid


        lambdaDotUpdater.writeParticlesData(); // DasteXar to write ParticlesData.txt in each time step for each rank





        // Momentum predictor

        fvVectorMatrix UEqn
        (
            fvm::ddt(U)
          + fvm::div(phi, U)
          - fvm::laplacian(nu, U)
           ==uSource
          );



        if (piso.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }

        // --- PISO loop
        while (piso.correct())
        {
            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
            surfaceScalarField phiHbyA
            (
                "phiHbyA",
                fvc::flux(HbyA)
              + fvc::interpolate(rAU)*fvc::ddtCorr(U, phi)
            );

            adjustPhi(phiHbyA, U, p);

            // Update the pressure BCs to ensure flux consistency
            constrainPressure(p, U, phiHbyA, rAU);

            // Non-orthogonal pressure corrector loop
            while (piso.correctNonOrthogonal())
            {
                // Pressure corrector

                fvScalarMatrix pEqn
                (
                    fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
                );

                pEqn.setReference(pRefCell, pRefValue);

            #if (newFoundationVersion)
              pEqn.solve();
            #else
              pEqn.solve(mesh.solver(p.select(piso.finalInnerIter())));
            #endif

                if (piso.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }


            #include "continuityErrs.H"

            U = HbyA - rAU*fvc::grad(p);
            U.correctBoundaryConditions();


        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
	yadeCoupling.setSourceZero();

    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
