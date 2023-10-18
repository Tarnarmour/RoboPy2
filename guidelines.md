# Code Guidelines

## Objectives

1. Prefer pure python implementations, because the major utility of this project is ease of use and low-complexity setup.
2. After (1), prefer easy-to-use and flexible API that works for many different types of inputs (e.g., accept input joint values as any iterable)
3. After (2), prefer high speed execution 
4. Prefer fewer layers of abstraction and avoid unnecessary class structure (e.g., no se3 class since numpy already works)

## Style Guide

1. Follow generic PEP guidelines
2. Prefer short and concise publicly facing function handles (e.g., fk instead of forward_kinematics)
3. Private functions should be more descriptive as necessary

## Naming Guidelines

1. q for joint values
2. x for cartesian values
3. A and T for poses
4. R for rotation matrices
