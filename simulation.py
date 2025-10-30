
import pygame
import numpy as np
import random
import os
import shutil
import sys
from bodies import Body
from quadtree import Quadtree
from itertools import combinations 

# --- CONSTANTS ---

# PYGAME
WIDTH, HEIGHT = 1600, 900
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)

# PHYSICS
G = 6.67430e-11
AU = 1.496e11  # Astronomical Unit (meters)
SOLAR_MASS = 1.989e30
TIMESTEP = 3600 * 24  # 1 day in seconds
BARNES_HUT_THETA = 0.5 # Accuracy of Barnes-Hut. 0 = direct, > 1 = less accurate


class Simulation:
    """
    Main class to run the N-body simulation, handle rendering, and user input.
    """

    def __init__(self):
        """Initializes the simulation, Pygame, and sets up the bodies."""
        pygame.init()
        pygame.display.set_caption("N-Body Galaxy Simulator")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        
        self.running = True
        self.paused = False
        self.total_time = 0
        self.selected_body = None
        self.use_barnes_hut = True
        self.show_quadtree = False
        
        # Camera
        self.SCALE = 250 / AU  # Initial scale: 250 pixels per AU
        self.offset = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        
        self.bodies = []
        self.quadtree = None
        
        # --- PRESET: Simple Solar System ---
        sun = Body(mass=SOLAR_MASS, 
                   position=[0, 0], 
                   velocity=[0, 0], 
                   color=(255, 255, 0), 
                   radius=6.96e8, 
                   name="Sun")

        earth = Body(mass=5.972e24, 
                     position=[AU, 0], 
                     velocity=[0, 29783], 
                     color=(0, 100, 255), 
                     radius=6.37e6, 
                     name="Earth")
                     
        self.bodies.append(sun)
        self.bodies.append(earth)
        
        # Center the camera on the most massive body (the Sun)
        self.offset = np.array([WIDTH / 2.0, HEIGHT / 2.0]) - sun.position * self.SCALE


    def run(self):
        """Main simulation loop."""
        while self.running:
            self._handle_input()
            
            if not self.paused:
                self._update_bodies(TIMESTEP)
                self._handle_collisions()

            self._draw()
            self.clock.tick(60)

    def _handle_input(self):
        """Handles user input for quitting, pausing, and camera controls."""
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_b:
                    self.use_barnes_hut = not self.use_barnes_hut
                if event.key == pygame.K_t:
                    self.show_quadtree = not self.show_quadtree
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zooming
                world_pos_before = (mouse_pos - self.offset) / self.SCALE
                if event.y > 0:
                    self.SCALE *= 1.1
                else:
                    self.SCALE /= 1.1
                world_pos_after = (mouse_pos - self.offset) / self.SCALE
                self.offset += (world_pos_after - world_pos_before) * self.SCALE

        # Panning
        if pygame.mouse.get_pressed()[0]:  # Left-click held
            rel = pygame.mouse.get_rel()
            self.offset += np.array([float(rel[0]), float(rel[1])])

    def _draw(self):
        """Draws all bodies, trails, and UI elements to the screen."""
        self.screen.fill(BLACK)

        # Draw Quadtree (if enabled)
        if self.use_barnes_hut and self.show_quadtree and self.quadtree:
            self._draw_quadtree(self.quadtree)

        # Draw trails and bodies
        for body in self.bodies:
            # 1. Draw Trail
            if len(body.trail) > 2:
                scaled_trail = [
                    (self.offset + pos * self.SCALE).astype(int) for pos in body.trail
                ]
                pygame.draw.lines(self.screen, body.color, False, scaled_trail, 1)

            # 2. Draw Body
            pos_x, pos_y = (self.offset + body.position * self.SCALE).astype(int)
            
            # Simple scaling for radius (not realistic, but visible)
            screen_radius = max(2, int(body.radius * self.SCALE * 500)) 
            
            # Clamp radius and position to prevent Pygame errors
            screen_radius = min(screen_radius, 200)
            if -1000 < pos_x < WIDTH + 1000 and -1000 < pos_y < HEIGHT + 1000:
                # Draw merge flash
                if body.merge_flash_timer > 0:
                    flash_color = (255, 255, 255)
                    flash_radius = screen_radius + (31 - body.merge_flash_timer) // 3
                    pygame.draw.circle(self.screen, flash_color, (pos_x, pos_y), flash_radius, 1)
                    body.merge_flash_timer -= 1
                
                pygame.draw.circle(self.screen, body.color, (pos_x, pos_y), screen_radius)

        # Draw UI / Info Text
        info_text = [
            f"FPS: {self.clock.get_fps():.1f}",
            f"Time: {self.total_time / (3600 * 24 * 365.25):.2f} years",
            f"Bodies: {len(self.bodies)}",
            f"Mode: {'Barnes-Hut (B)' if self.use_barnes_hut else 'Direct Sum (B)'}",
            f"Quadtree: {'Visible (T)' if self.show_quadtree else 'Hidden (T)'}",
            f"Paused: {self.paused} (P)"
        ]
        for i, line in enumerate(info_text):
            text_surf = self.font.render(line, True, WHITE)
            self.screen.blit(text_surf, (10, 10 + i * 20))

        pygame.display.flip()
        
    def _draw_quadtree(self, node):
        """draws the quadtree boundaries."""
        if node is None:
            return
            
        x, y, w, h = node.bounds
        # Convert world coords to screen coords
        rect_x = int(x * self.SCALE + self.offset[0])
        rect_y = int(y * self.SCALE + self.offset[1])
        rect_w = int(w * self.SCALE)
        rect_h = int(h * self.SCALE)
        
        # Only draw if on screen
        if rect_x < WIDTH and rect_x + rect_w > 0 and rect_y < HEIGHT and rect_y + rect_h > 0:
            pygame.draw.rect(self.screen, (50, 50, 50), (rect_x, rect_y, rect_w, rect_h), 1)

            if any(node.children):
                for child in node.children:
                    self._draw_quadtree(child)

    def _recalculate_all_accelerations(self):
        """Calculates acceleration for all bodies using Barnes-Hut or Direct Sum."""
        
        # Reset accelerations
        for body in self.bodies:
            body.acceleration.fill(0.0)

        if self.use_barnes_hut:
            # --- Barnes-Hut O(n log n) ---
            
            # Find bounds
            min_pos = np.min([b.position for b in self.bodies], axis=0)
            max_pos = np.max([b.position for b in self.bodies], axis=0)
            center = (min_pos + max_pos) / 2.0
            size = np.max(max_pos - min_pos) * 1.1 # Add 10% padding
            if size == 0: size = 2 * AU # Handle single-body case
            
            # Build quadtree
            self.quadtree = Quadtree(center[0] - size / 2, center[1] - size / 2, size, size)
            for body in self.bodies:
                self.quadtree.insert(body)
                
            # Calculate forces
            for body in self.bodies:
                body.acceleration = self._calculate_force(body, self.quadtree)
        
        else:
            # --- Direct Sum O(n^2) ---
            self.quadtree = None # No quadtree in this mode
            for body_a, body_b in combinations(self.bodies, 2):
                r_vec = body_b.position - body_a.position
                dist_sq = np.dot(r_vec, r_vec)
                
                # Avoid division by zero if bodies are in the same spot
                if dist_sq == 0:
                    continue
                    
                dist = np.sqrt(dist_sq)
                
                # F = G * m1 * m2 / r^2
                force_mag = (G * body_a.mass * body_b.mass) / dist_sq
                force_vec = force_mag * (r_vec / dist)
                
                # a = F / m
                body_a.acceleration += force_vec / body_a.mass
                body_b.acceleration -= force_vec / body_b.mass

    def _calculate_force(self, body, node):
        """
        calculates the force on a body using the Barnes-Hut logic.
        """
        if node is None or node.total_mass == 0:
            return np.array([0.0, 0.0])

        # Case 1: Node is an external node (leaf) with a body
        if node.body is not None:
            # Don't calculate force with self
            if node.body is body:
                return np.array([0.0, 0.0])
            
            # Direct calculation
            r_vec = node.body.position - body.position
            dist_sq = np.dot(r_vec, r_vec)
            if dist_sq == 0:
                return np.array([0.0, 0.0])
            dist = np.sqrt(dist_sq)
            force_mag = (G * body.mass * node.body.mass) / dist_sq
            force_vec = force_mag * (r_vec / dist)
            return force_vec / body.mass

        # Case 2: Node is an internal node
        
        # Calculate s/d
        s = node.bounds[2]  # Width of the node's region
        r_vec = node.center_of_mass - body.position
        d_sq = np.dot(r_vec, r_vec)
        if d_sq == 0:
            return np.array([0.0, 0.0]) # Body is at the center of mass
        
        d = np.sqrt(d_sq)

        if (s / d) < BARNES_HUT_THETA:
            # Case 2a: Node is far enough away. Treat as a single mass.
            force_mag = (G * body.mass * node.total_mass) / d_sq
            force_vec = force_mag * (r_vec / d)
            return force_vec / body.mass
        else:
            # Case 2b: Node is too close. Recurse into children.
            total_acceleration = np.array([0.0, 0.0])
            for child in node.children:
                total_acceleration += self._calculate_force(body, child)
            return total_acceleration


 
    def _update_bodies(self, dt):
        """
        Updates body positions and velocities using the Velocity Verlet integrator.
    
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

    def _get_bodies_in_node_recursive(self, node, body_list):
        """A helper to recursively collect all bodies within a given quadtree node."""
        if node is None:
            return
        if node.body is not None:
            body_list.append(node.body)
        for child in node.children:
            self._get_bodies_in_node_recursive(child, body_list)

    def _handle_collisions(self):
        """
        Handles collisions efficiently using the quadtree to only check nearby bodies.
        """
        bodies_to_remove = set()
        
        # In Barnes-Hut mode, use the quadtree to find collision pairs.
        if self.use_barnes_hut and self.quadtree:
            checked_pairs = set()

            def find_and_resolve(node):
                if node is None or node.total_mass == 0:
                    return

                if any(node.children):
                    bodies_in_node = []
                    self._get_bodies_in_node_recursive(node, bodies_in_node)

                    for body_a, body_b in combinations(bodies_in_node, 2):
                        pair_key = tuple(sorted((id(body_a), id(body_b))))
                        if pair_key in checked_pairs:
                            continue
                        
                        checked_pairs.add(pair_key)
                        
                        if body_a not in bodies_to_remove and body_b not in bodies_to_remove:
                            # --- CRITICAL FIX APPLIED ---
                            if np.linalg.norm(body_a.position - body_b.position) < (body_a.radius + body_b.radius):
                                absorber, absorbed = (body_a, body_b) if body_a.mass > body_b.mass else (body_b, body_a)
                                new_vel = (absorber.mass * absorber.velocity + absorbed.mass * absorbed.velocity) / (absorber.mass + absorbed.mass)
                                absorber.mass += absorbed.mass
                                absorber.velocity = new_vel
                                absorber.radius = (absorber.radius**3 + absorbed.radius**3)**(1/3) # Combine volume
                                absorber.merge_flash_timer = 30
                                bodies_to_remove.add(absorbed)
                    
                    for child in node.children:
                        find_and_resolve(child)
            
            find_and_resolve(self.quadtree)

        # Fallback to O(n^2) direct-sum mode if not using Barnes-Hut
        else:
            for i, body_a in enumerate(self.bodies):
                if body_a in bodies_to_remove: continue
                for body_b in self.bodies[i+1:]:
                    if body_b in bodies_to_remove: continue
              
                    if np.linalg.norm(body_a.position - body_b.position) < (body_a.radius + body_b.radius):
                        absorber, absorbed = (body_a, body_b) if body_a.mass > body_b.mass else (body_b, body_a)
                        new_vel = (absorber.mass * absorber.velocity + absorbed.mass * absorbed.velocity) / (absorber.mass + absorbed.mass)
                        absorber.mass += absorbed.mass
                        absorber.velocity = new_vel
                        absorber.radius = (absorber.radius**3 + absorbed.radius**3)**(1/3) # Combine volume
                        absorber.merge_flash_timer = 30
                        bodies_to_remove.add(absorbed)

        # Update the main bodies list after all checks are done
        if bodies_to_remove:
            if self.selected_body in bodies_to_remove:
                self.selected_body = None
            self.bodies = [b for b in self.bodies if b not in bodies_to_remove]