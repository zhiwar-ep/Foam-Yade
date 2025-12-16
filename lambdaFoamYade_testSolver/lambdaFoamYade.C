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



	yadeCoupling.locateAllParticles();   // places the location of spheres from time 0



    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        #include "CourantNo.H"


        vGrad = fvc::grad(U);
        
       

        yadeCoupling.setParticleAction(runTime.deltaT().value());
        


        

//  DasteXar Update lambdaDot with constant rate

	//lambdaDot += (0.1 * runTime.deltaT().value());  // same for all cells
	
	forAll(lambdaDot, i)
		{
		    const point& c = mesh.C()[i];
		    lambdaDot[i] = 0.1 * Foam::sin(c.x());   // or something else spatially varying
		}


	lambdaDot.correctBoundaryConditions();



  
// DasteXar  Count particles per cell 
	nParticles = 0.0;

	for (const auto& procPtr : yadeCoupling.inCommProcs)
	{
	    if (!procPtr) continue;

	    for (const auto& partPtr : procPtr->foundParticles)
	    {
		if (!partPtr) continue;

		const point& center = partPtr->pos;
		label cellI = mesh.findCell(center);

		if (cellI >= 0)
		    nParticles[cellI] += 1.0;
	    }
	}
	
	
//DasteXar
// Assign lambdaDot of each cell to spheres inside that cell
for (const auto& procPtr : yadeCoupling.inCommProcs)
{
    if (!procPtr) continue;

    for (const auto& partPtr : procPtr->foundParticles)
    {
        if (!partPtr) continue;

        const point& center = partPtr->pos;
        label cellI = mesh.findCell(center);

        // Skip empty cells
        if (cellI < 0) continue;
        if (nParticles[cellI] < 0.5) continue;

        // Read   lambdaDot and assign it to the particle
        scalar lambdaDotVal = lambdaDot[cellI];
        partPtr->lambdaDot = lambdaDotVal;
    }
}





//  non-zero lambdaDot particles: (particleID, cellID, lambdaDot)
// one file per MPI rank to avoid contention; append each timestep

// NOTE: this particleID is different with particleID of yade !!! 

if (runTime.outputTime())

{
    const int myRank = Pstream::myProcNo();

    // build per-rank path in the current time folder
    fileName outDir = runTime.timePath();  
    mkDir(outDir);                         

    fileName outPath = outDir / ("ParticlesData.txt");

    std::ofstream ofs(outPath.c_str(), std::ios::app);
    ofs.setf(std::ios::scientific);
    ofs.precision(8);

    ofs << "# time " << runTime.timeName() << "  (particleID  cellID  lambdaDot)\n";

    for (const auto& procPtr : yadeCoupling.inCommProcs)
    {
        if (!procPtr) continue;

        for (const auto& partPtr : procPtr->foundParticles)
        {
            if (!partPtr) continue;

            const point& center = partPtr->pos;
            label cellI = mesh.findCell(center);

            if (cellI < 0) continue;
            if (nParticles[cellI] < 0.5) continue;

            const scalar ld = lambdaDot[cellI];

            ofs << partPtr->indx << "  " << cellI << "  " << ld << "\n";
        }
    }

    ofs << "\n"; 
}



        
        


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
