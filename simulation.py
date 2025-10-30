
# At the top of simulation.py, add this import
import pygame
import numpy as np
import random
import os
import shutil
from bodies import Body
from quadtree import Quadtree
from itertools import combinations # <--- ADD THIS LINE

AU = 1.496e11
# ... (the rest of your imports and constants) ...


class Simulation:
    # ... (your __init__ and other methods are fine) ...

    # REPLACE this method in your Simulation class
    def _update_bodies(self, dt):
        """
        Updates body positions and velocities using the Velocity Verlet integrator.
        This is more efficient as it only requires one force calculation per step.
        """
        self.total_time += dt

        # 1. Update positions based on current velocity and acceleration
        for body in self.bodies:
            body.position += body.velocity * dt + 0.5 * body.acceleration * dt * dt
            body.trail.append(body.position.copy())

        # 2. Store the old accelerations before recalculating
        old_accelerations = {body: body.acceleration for body in self.bodies}

        # 3. Calculate the NEW accelerations at the new positions (ONLY ONCE)
        self._recalculate_all_accelerations()

        # 4. Update velocities using the average of the old and new accelerations
        for body in self.bodies:
            # Check if the body is still in the simulation (wasn't merged)
            if body in old_accelerations:
                body.velocity += 0.5 * (old_accelerations[body] + body.acceleration) * dt

    # ADD this new helper method inside your Simulation class
    def _get_bodies_in_node_recursive(self, node, body_list):
        """A helper to recursively collect all bodies within a given quadtree node."""
        if node is None:
            return
        if node.body is not None:
            body_list.append(node.body)
        for child in node.children:
            self._get_bodies_in_node_recursive(child, body_list)

    # REPLACE this method in your Simulation class
    def _handle_collisions(self):
        """
        Handles collisions efficiently using the quadtree to only check nearby bodies.
        """
        bodies_to_remove = set()
        
        # In Barnes-Hut mode, use the quadtree to find collision pairs.
        if self.use_barnes_hut and self.quadtree:
            # A set to keep track of pairs we've already checked to avoid redundant work
            checked_pairs = set()

            # The recursive function will find and resolve collisions
            def find_and_resolve(node):
                if node is None or node.total_mass == 0:
                    return

                # If a node is an internal node (has children), it means there's a group of
                # bodies close together. We check for collisions within this group.
                if any(node.children):
                    bodies_in_node = []
                    self._get_bodies_in_node_recursive(node, bodies_in_node)

                    # Check all unique pairs of bodies within this node
                    for body_a, body_b in combinations(bodies_in_node, 2):
                        # Use IDs to create a unique, ordered key for the pair
                        pair_key = tuple(sorted((id(body_a), id(body_b))))
                        if pair_key in checked_pairs:
                            continue
                        
                        checked_pairs.add(pair_key)
                        
                        if body_a not in bodies_to_remove and body_b not in bodies_to_remove:
                            if np.linalg.norm(body_a.position - body_b.position) < (body_a.radius + body_b.radius) / self.SCALE:
                                absorber, absorbed = (body_a, body_b) if body_a.mass > body_b.mass else (body_b, body_a)
                                new_vel = (absorber.mass * absorber.velocity + absorbed.mass * absorbed.velocity) / (absorber.mass + absorbed.mass)
                                absorber.mass += absorbed.mass
                                absorber.velocity = new_vel
                                absorber.radius = (absorber.radius**3 + absorbed.radius**3)**(1/3)
                                absorber.merge_flash_timer = 30
                                bodies_to_remove.add(absorbed)
                    
                    # Continue searching for smaller, denser groups in the children
                    for child in node.children:
                        find_and_resolve(child)
            
            find_and_resolve(self.quadtree)

        # Fallback to O(n^2) direct-sum mode if not using Barnes-Hut
        else:
            for i, body_a in enumerate(self.bodies):
                if body_a in bodies_to_remove: continue
                for body_b in self.bodies[i+1:]:
                    if body_b in bodies_to_remove: continue
                    if np.linalg.norm(body_a.position - body_b.position) < (body_a.radius + body_b.radius) / self.SCALE:
                        absorber, absorbed = (body_a, body_b) if body_a.mass > body_b.mass else (body_b, body_a)
                        new_vel = (absorber.mass * absorber.velocity + absorbed.mass * absorbed.velocity) / (absorber.mass + absorbed.mass)
                        absorber.mass += absorbed.mass
                        absorber.velocity = new_vel
                        absorber.radius = (absorber.radius**3 + absorbed.radius**3)**(1/3)
                        absorber.merge_flash_timer = 30
                        bodies_to_remove.add(absorbed)

        # Update the main bodies list after all checks are done
        if bodies_to_remove:
            if self.selected_body in bodies_to_remove:
                self.selected_body = None
            self.bodies = [b for b in self.bodies if b not in bodies_to_remove]


# Make sure the rest of your Simulation class and other files remain the same.