# main.py
import pygame
from simulation import Simulation

def main():
    """Main function to run the simulation."""
    pygame.init()

    # Create a simulation instance
    sim = Simulation()

    # Start the main simulation loop
    sim.run()

    pygame.quit()

if __name__ == "__main__":
    main()