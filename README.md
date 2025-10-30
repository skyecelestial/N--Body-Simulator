# 🌌 N-Body Galaxy Simulator

![Demo of the Galaxy Simulator](demo.gif)

This project is a 2D N-body gravity simulator written in Python using Pygame. It visualizes the gravitational interactions of many bodies in real-time.

The core feature is the implementation of the **Barnes-Hut algorithm**, an approximation method that scales to thousands of bodies. This allows the simulation to run in **O(n log n)** time, a massive improvement over the **O(n²)** time required by the (also included) direct-sum method.

### How It Works

* **Quadtree:** The 2D space is recursively partitioned into a  Quadtree.
* **Barnes-Hut:** For any given body, the gravitational force from distant groups of bodies is approximated as a single force from their combined center of mass. This avoids thousands of individual calculations.
* **Physics:** The simulation uses the **Velocity Verlet integrator** for a stable and accurate numerical integration of motion.
* **Collisions:** Bodies are treated as inelastic. When they collide, they merge, conserving momentum and combining their mass and volume.

---

##  Features

* **Interactive Camera:** Smoothly **pan** (left-click + drag) and **zoom** (mouse wheel) to explore the universe.
* **Physics Engine:**
    * Stable Velocity Verlet integration.
    * Inelastic collisions with momentum conservation.
* **Real-time Debugging:**
    * Toggle the **Quadtree visualization** to see how the space is being partitioned.
    * Live UI overlay shows FPS, body count, simulation time, and current mode.

---

### Prerequisites

You must have Python 3 and `pip` installed.

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the simulation!**
    ```bash
    python main.py
    ```

---



