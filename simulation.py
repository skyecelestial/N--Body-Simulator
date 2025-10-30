import pygame
import numpy as np
import random
import os
import shutil
from bodies import Body
from quadtree import Quadtree

AU = 1.496e11
G = 39.478
TIMESTEP_BASE = 1 / 365.25

class Simulation:
    def __init__(self, width=800, height=800):
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("N-Body Simulator")
        self.clock = pygame.time.Clock()
        self.bodies = []
        self.SCALE = 150
        self.OFFSET = np.array([width / 2, height / 2])
        self.panning = False
        self.paused = False
        self.timestep = TIMESTEP_BASE
        self.total_time = 0.0
        self.sub_steps = 1
        self.spawning = False
        self.spawn_pos = None
        self.mouse_pos = None
        self.VELOCITY_SCALE = 0.05
        pygame.font.init()
        self.font = pygame.font.SysFont("Segoe UI", 20)
        self.trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.selected_body = None
        self.use_barnes_hut = False
        self.theta = 0.5
        self.show_quadtree = False
        self.quadtree = None
        self._setup_initial_conditions()

    def _setup_initial_conditions(self):
        self.bodies.clear()
        sun = Body(mass=1.0, position=[0, 0], velocity=[0, 0], color=(255, 255, 0), radius=15, name="Sun")
        self.bodies.append(sun)
        self.selected_body = None
        self.total_time = 0.0

    def _generate_galaxy(self, num_bodies=500):
        self._setup_initial_conditions()
        sun = self.bodies[0]
        for _ in range(num_bodies):
            dist = random.uniform(2, 15)
            angle = random.uniform(0, 2 * np.pi)
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            circular_speed = np.sqrt(G * sun.mass / dist)
            orbital_speed = circular_speed * random.uniform(0.7, 1.3)
            vel = orbital_speed * np.array([-np.sin(angle), np.cos(angle)])
            if random.random() < 0.05:
                mass = random.uniform(1e-5, 5e-5)
                radius = random.randint(8, 12)
                color = (255, random.randint(100, 150), 0)
            else:
                mass = random.uniform(1e-7, 1e-6)
                radius = random.randint(2, 5)
                color = (random.randint(100, 200), random.randint(100, 200), random.randint(200, 255))
            self.bodies.append(Body(mass, pos, vel, color, radius))
        self.SCALE = 40

    def _calculate_force_direct(self, body_a):
        total_force = np.array([0, 0], dtype=float)
        for body_b in self.bodies:
            if body_a == body_b: continue
            r_vec = body_b.position - body_a.position
            dist = np.linalg.norm(r_vec)
            epsilon = 1e-6 
            force_magnitude = G * body_a.mass * body_b.mass / (dist**2 + epsilon)
            force_vector = force_magnitude * (r_vec / dist)
            total_force += force_vector
        return total_force

    def _calculate_force_barnes_hut(self, body, node):
        if node is None or node.total_mass == 0: return np.array([0.0, 0.0])
        if node.body is not None:
            if node.body is not body:
                r_vec = node.body.position - body.position
                dist = np.linalg.norm(r_vec)
                epsilon = 1e-6
                force_magnitude = G * body.mass * node.body.mass / (dist**2 + epsilon)
                return force_magnitude * (r_vec / dist)
            else: return np.array([0.0, 0.0])
        s = node.bounds[2]
        d = np.linalg.norm(node.center_of_mass - body.position)
        if d > 0 and s / d < self.theta:
            r_vec = node.center_of_mass - body.position
            dist_sq = d**2
            epsilon = 1e-6
            force_magnitude = G * body.mass * node.total_mass / (dist_sq + epsilon)
            return force_magnitude * (r_vec / d)
        else:
            total_force = np.array([0.0, 0.0])
            for child in node.children:
                if child: total_force += self._calculate_force_barnes_hut(body, child)
            return total_force

    def _recalculate_all_accelerations(self):
        if self.use_barnes_hut and len(self.bodies) > 1:
            min_pos = np.min([b.position for b in self.bodies], axis=0)
            max_pos = np.max([b.position for b in self.bodies], axis=0)
            size = np.max(max_pos - min_pos) * 1.1
            center = (min_pos + max_pos) / 2
            self.quadtree = Quadtree(center[0] - size/2, center[1] - size/2, size, size)
            for body in self.bodies: self.quadtree.insert(body)
            for body in self.bodies:
                force = self._calculate_force_barnes_hut(body, self.quadtree)
                body.acceleration = force / body.mass
        else:
            self.quadtree = None
            for body in self.bodies:
                force = self._calculate_force_direct(body)
                body.acceleration = force / body.mass

    def _update_bodies(self, dt):
        self.total_time += dt
        self._recalculate_all_accelerations()
        for body in self.bodies: body.velocity += 0.5 * body.acceleration * dt
        for body in self.bodies:
            body.position += body.velocity * dt
            body.trail.append(body.position.copy())
        self._recalculate_all_accelerations()
        for body in self.bodies: body.velocity += 0.5 * body.acceleration * dt
            
    def _spawn_body(self, end_pos):
        start_au = (np.array(self.spawn_pos) - self.OFFSET) / self.SCALE
        velocity = (np.array(end_pos) - self.spawn_pos) * self.VELOCITY_SCALE
        mass = random.uniform(1e-6, 1e-4)
        radius = random.randint(3, 8)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.bodies.append(Body(mass, start_au, velocity, color, radius))

    def _handle_collisions(self):
        bodies_to_remove = set()
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
        if bodies_to_remove:
            if self.selected_body in bodies_to_remove: self.selected_body = None
            self.bodies = [b for b in self.bodies if b not in bodies_to_remove]

    def _draw_bodies(self):
        for body in self.bodies:
            screen_pos = body.position * self.SCALE + self.OFFSET
            draw_radius = max(body.radius * (self.SCALE / 150)**0.5, 1)
            if body.merge_flash_timer > 0:
                p = body.merge_flash_timer / 30
                color = tuple(int(c + (255 - c) * p) for c in body.color)
                pygame.draw.circle(self.screen, color, screen_pos.astype(int), int(draw_radius))
                body.merge_flash_timer -= 1
            else:
                pygame.draw.circle(self.screen, body.color, screen_pos.astype(int), int(draw_radius))
            if body == self.selected_body:
                pygame.draw.circle(self.screen, (255, 255, 255), screen_pos.astype(int), int(draw_radius) + 4, 2)

    def _draw_trails(self):
        self.trail_surface.fill((0,0,0,0))
        for body in self.bodies:
            if len(body.trail) < 2: continue
            points = [p * self.SCALE + self.OFFSET for p in body.trail]
            for i in range(len(points) - 1):
                alpha = 255 * (i / len(points))
                color = (*body.color, alpha)
                pygame.draw.line(self.trail_surface, color, points[i], points[i+1], max(1, int(2 * self.SCALE / 150)))
        self.screen.blit(self.trail_surface, (0,0))

    def _draw_quadtree(self, node):
        if node is None: return
        rect = pygame.Rect(node.bounds[0] * self.SCALE + self.OFFSET[0], node.bounds[1] * self.SCALE + self.OFFSET[1],
                           node.bounds[2] * self.SCALE, node.bounds[3] * self.SCALE)
        pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        for child in node.children: self._draw_quadtree(child)

    def _draw_ui(self):
        time_str = f"{self.total_time / 1e6:.3f} mYears" if self.total_time >= 1e6 else \
                   f"{self.total_time / 1e3:.2f} kYears" if self.total_time >= 1e3 else \
                   f"{self.total_time:.1f} Years"
        self.screen.blit(self.font.render(f"Elapsed Time: {time_str}", True, (255, 255, 255)), (10, 60))
        self.screen.blit(self.font.render(f"Bodies: {len(self.bodies)}", True, (255, 255, 255)), (10, 10))
        self.screen.blit(self.font.render(f"Time Multiplier (+/-): {self.timestep / TIMESTEP_BASE:.1f}x", True, (255, 255, 255)), (10, 35))
        mode_str = "Barnes-Hut (O(nlogn))" if self.use_barnes_hut else "Direct Sum (O(n^2))"
        mode_color = (0, 255, 150) if self.use_barnes_hut else (255, 150, 0)
        self.screen.blit(self.font.render(f"Mode (B): {mode_str}", True, mode_color), (10, 85))
        if self.use_barnes_hut:
            self.screen.blit(self.font.render(f"Theta (↑↓): {self.theta:.2f}", True, (255, 255, 255)), (10, 110))
        self.screen.blit(self.font.render(f"Physics Steps/Frame: {self.sub_steps}", True, (255, 255, 255)), (10, 135))
        self.screen.blit(self.font.render("G: Gen Galaxy", True, (200, 200, 200)), (10, self.height - 55))
        self.screen.blit(self.font.render("F/S: Fast Forward/Slow", True, (200, 200, 200)), (10, self.height - 30))
        if self.paused:
            p_text = self.font.render("PAUSED", True, (255, 200, 0))
            self.screen.blit(p_text, p_text.get_rect(center=(self.width / 2, 25)))
            
    def _draw_inspector(self):
        if not self.selected_body: return
        b = self.selected_body
        name = b.name or "Unnamed"
        panel = pygame.Rect(self.width - 260, 10, 250, 120)
        pygame.draw.rect(self.screen, (20, 20, 40, 200), panel, border_radius=10)
        pygame.draw.rect(self.screen, (100, 100, 120), panel, 2, border_radius=10)
        self.screen.blit(self.font.render(f"INSPECTING: {name}", True, (255, 255, 0)), (panel.x + 10, panel.y + 5))
        self.screen.blit(self.font.render(f"Mass: {b.mass:.3e} (Solar)", True, (255, 255, 255)), (panel.x + 10, panel.y + 35))
        self.screen.blit(self.font.render(f"Pos: [{b.position[0]:.2f}, {b.position[1]:.2f}] AU", True, (255, 255, 255)), (panel.x + 10, panel.y + 60))
        self.screen.blit(self.font.render(f"Vel: [{b.velocity[0]:.2f}, {b.velocity[1]:.2f}] AU/yr", True, (255, 255, 255)), (panel.x + 10, panel.y + 85))

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEWHEEL:
                    m_pos = (np.array(pygame.mouse.get_pos()) - self.OFFSET) / self.SCALE
                    self.SCALE *= 1.1 if event.y > 0 else 1/1.1
                    self.OFFSET = -m_pos * self.SCALE + pygame.mouse.get_pos()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    if event.key == pygame.K_c: self._setup_initial_conditions()
                    if event.key in (pygame.K_EQUALS, pygame.K_PLUS): self.timestep *= 2.0
                    if event.key == pygame.K_MINUS: self.timestep /= 2.0
                    if event.key == pygame.K_f: self.timestep *= 10.0
                    if event.key == pygame.K_s: self.timestep /= 10.0
                    if event.key == pygame.K_b: self.use_barnes_hut = not self.use_barnes_hut
                    if event.key == pygame.K_q: self.show_quadtree = not self.show_quadtree
                    if event.key == pygame.K_UP: self.theta = min(2.0, self.theta + 0.05)
                    if event.key == pygame.K_DOWN: self.theta = max(0.0, self.theta - 0.05)
                    if event.key == pygame.K_g: self._generate_galaxy()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: self.spawning, self.spawn_pos = True, pygame.mouse.get_pos()
                    if event.button == 2: self.panning = True; self.selected_body = None
                    if event.button == 3:
                        m_pos = (np.array(event.pos) - self.OFFSET) / self.SCALE
                        clicked = False
                        for b in reversed(self.bodies):
                            if np.linalg.norm(b.position - m_pos) < b.radius / self.SCALE:
                                self.selected_body = b if self.selected_body != b else None
                                clicked = True; break
                        if not clicked: self.selected_body = None
                if event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = pygame.mouse.get_pos()
                    if self.panning:
                        self.OFFSET += event.rel
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.spawning: self._spawn_body(pygame.mouse.get_pos()); self.spawning = False
                    if event.button == 2: self.panning = False
            if not self.paused:
                self.sub_steps = int(1 + self.timestep / (TIMESTEP_BASE * 50))
                sub_dt = self.timestep / self.sub_steps
                for _ in range(self.sub_steps):
                    self._update_bodies(sub_dt)
                    self._handle_collisions()
            if self.selected_body and not self.panning:
                target = np.array([self.width / 2, self.height / 2])
                current = self.selected_body.position * self.SCALE + self.OFFSET
                self.OFFSET += target - current
            self.screen.fill((0, 0, 10))
            if self.show_quadtree and self.quadtree: self._draw_quadtree(self.quadtree)
            self._draw_trails()
            self._draw_bodies()
            if self.spawning and self.mouse_pos is not None:
                pygame.draw.line(self.screen, (255, 255, 255), self.spawn_pos, self.mouse_pos, 2)
            self._draw_ui()
            self._draw_inspector()
            pygame.display.flip()