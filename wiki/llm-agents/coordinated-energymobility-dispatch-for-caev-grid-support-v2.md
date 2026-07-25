# Coordinated Energy–Mobility Dispatch for CAEV Grid Support (V2G)

## Summary
This paper proposes a coordinated energy–mobility dispatch framework that routes a fleet of Connected Autonomous Electric Vehicles (CAEVs) equipped with virtual battery partitioning toward Vehicle-to-Grid (V2G) stations to meet a Distributed System Operator's (DSO) request for a specified amount of energy within a deadline. The framework combines a traffic-aware routing layer with a vehicle-level speed-control layer to guarantee timely, feasible delivery of grid support energy despite congestion. It is validated through simulation on the real urban network of Rapallo, Italy.

## Key Ideas
- Two-layer decision architecture: an upper routing layer and a lower vehicle-control layer.
- Virtual Battery Partitioning (VBP) splits each CAEV's onboard battery into a mobility partition and a Grid Support Service (GSS) partition, preserving driving needs while enabling reliable service provision.
- Upper layer solves a periodically updated Resource-Constrained Shortest Path (RCSP) problem, treating time and energy as constrained resources to satisfy both the DSO deadline and the required grid-support energy.
- Routing costs are updated using the state of a dynamic macroscopic traffic model, making dispatch congestion-aware and adaptive.
- Lower layer uses a Model Predictive Control (MPC) strategy to regulate each vehicle's speed along its assigned path, enforcing mobility energy constraints and ensuring arrival within the required time window.
- Targets a gap in prior V2G/EV-dispatch literature, which generally neglects the interaction between energy service provision and traffic dynamics (e.g., congestion-induced delays).

## Method
The urban area is modeled as a directed graph 𝒢 = (𝒱, ℰ) with J junctions and M directed road links; a subset 𝒱_V2G ⊂ 𝒱 of junctions hosts bidirectional V2G stations, and the city is partitioned into D energy districts, each with its own subset of V2G nodes.

Traffic dynamics follow the macroscopic Urban Traffic Control (TUC) model. Each link (i,h) has a vehicle count x_i^h evolving via a conservation equation driven by inflow q_i^h, exit flow s_i^h, boundary inflow d_i^h, and outflow l_i^h (Eqs. 1–4). Junctions are split into right-of-way (ROW) junctions, where outflow is capped by a saturation flow rate Φ_i^h, and traffic-light (TL) junctions, where outflow additionally depends on signal stage green times (cycle time C, lost time H, effective green ḵg, shared across all TL junctions and stages). A turning-rate map ξ_j governs how flow from incoming to outgoing links is split, biased toward paths that shorten the distance to sink nodes and forbidding u-turns.

Each CAEV's battery is split by its Battery Management System into two virtual partitions with maximum capacities Ē_c,mob (mobility) and Ē_c,GSS (grid support), decoupling driving-range assurance from GSS commitments.

Dispatch proceeds as: the DSO issues a request for a defined amount of energy to be delivered within a time horizon; the upper layer periodically re-solves an RCSP over the traffic-aware graph — using time and energy as constrained resources — to assign each CAEV a route to a V2G station that meets both the deadline and energy requirement; the lower layer's MPC then regulates vehicle speed along that route to satisfy the mobility energy budget while ensuring on-time arrival.

## Results
The provided text does not include the numerical simulation results (Section IV, Rapallo case study, is not present in the excerpt). The abstract states that the framework was "validated through simulations on the urban network of Rapallo (Italy), demonstrating robustness against congestion-induced delays," but no concrete metrics are given in the available content.

## Why It Matters
Existing EV/CAEV dispatch and V2G scheduling approaches typically treat energy service provision separately from traffic dynamics, ignoring how congestion-related delays can jeopardize timely delivery of grid support services in real urban settings. By jointly optimizing traffic-aware routing (RCSP) and vehicle-level speed control (MPC) under a virtual battery partitioning scheme, this framework aims to make CAEV fleets a more predictable and reliable source of fast, distributed grid flexibility as RES penetration and EV electrification both grow.

---
Sources: Nikolas Sacchi, Giacomo Basile, Silvia Siri, Manuela Minetti, Andrea Bonfiglio, Antonella Ferrara; arXiv (eess.SY), 25 May 2026
Raw: https://arxiv.org/html/2605.25847
Updated: 2026-07-24
