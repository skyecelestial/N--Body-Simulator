# main.py
import pygame
from simulation import Simulation

def main():
    """Run simulation"""
    pygame.init()

    sim = Simulation()

    sim.run()

    pygame.quit()

if __name__ == "__main__":
    main()