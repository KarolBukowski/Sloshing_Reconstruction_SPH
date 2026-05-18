# Sloshing_Reconstruction_SPH

This repository contains tools for sparse reconstruction of sloshing free-surface fields from SPH simulations of a rectangular tank.

The goal is to learn dominant spatial modes from all cases together and reconstruct the full free-surface elevation field from a small number of optimally placed gauges on the physical experimental grid.

Two reconstruction workflows are included:

## 2D reconstruction

`main_2D.py` performs sparse reconstruction of the 1D sloshing free-surface in a 2D rectangular tank.

SPH simulations of a rectangular sloshing tank with length \(0.45 \, \mathrm{m}\) are run at several forcing frequencies. Each case provides the free-surface elevation at many spatial points and time steps.

## 3D reconstruction

`main_3D.py` performs sparse reconstruction of the full 2D free-surface field in a rectangular tank.

SPH simulations of a rectangular sloshing tank with dimensions \(0.45 \, \mathrm{m} \times 0.30 \, \mathrm{m}\) are run at several single-axis \(X\), single-axis \(Y\), and combined \(X+Y\) forcing frequencies. Each case provides the free-surface elevation over a full 2D spatial grid at many time steps.

