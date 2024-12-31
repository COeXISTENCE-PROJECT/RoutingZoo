# RoutingZoo

**Routing ZOO** is a simulation platform where virtual drivers experiment with routing strategies to navigate from origins to destinations in dense urban networks. Participants engage in a "routing game," where their collective path choices generate congestion, influence expected travel costs, and adapt through learning and available information.

The framework defines scenarios using a network (based on OSM graphs) and demand patterns (players characterized by $(o_i, d_i, t_i)$), simulated over a fixed number of days (typically 300). Each day corresponds to a SUMO simulation, where agents' collective decisions determine actual travel costs, primarily travel times.

Each agent follows a predefined behavioral model, ranging from simple methods (e.g., Markov-Learning, Weighted-Average, Bounded Rationality) to complex, custom models implemented via Python scripts.

Upon completing the simulation, we provide detailed statistics at two levels:

- **Agent-Level**: Insights into individual learning processes, convergence speed, choice stability, and learning outcomes.
- **System-Level**: Evaluations of the network solution’s proximity to Wardrop Equilibrium, its stability over time, and its empirical realism.
