import numpy as np

class Quadtree:
    """
    Represents node in quadtree
    """
    def __init__(self, x, y, width, height):
        self.bounds = (x, y, width, height)
        self.body = None 
        self.children = [None, None, None, None] 
        
        self.total_mass = 0
        self.center_of_mass = np.array([0.0, 0.0])
        
    def _get_quadrant(self, body):
        """Decides which quadrant body belongs to"""
        x, y, width, height = self.bounds
        mid_x = x + width / 2
        mid_y = y + height / 2
        
        pos = body.position
        if pos[0] < mid_x:
            return 0 if pos[1] < mid_y else 2 
        else:
            return 1 if pos[1] < mid_y else 3 
            
    def _subdivide(self):
        """Subdivides the node into four children quadrants."""
        x, y, width, height = self.bounds
        half_w, half_h = width / 2, height / 2
        
        self.children[0] = Quadtree(x, y, half_w, half_h) 
        self.children[1] = Quadtree(x + half_w, y, half_w, half_h) 
        self.children[2] = Quadtree(x, y + half_h, half_w, half_h) 
        self.children[3] = Quadtree(x + half_w, y + half_h, half_w, half_h)

    def insert(self, body):
        """Inserts a body into the quadtree."""
        if self.body is None and not any(self.children):
            self.body = body
            self.total_mass = body.mass
            self.center_of_mass = body.position.copy()
            return

        
        if any(self.children):
            self._update_mass(body)
            quadrant_idx = self._get_quadrant(body)
            self.children[quadrant_idx].insert(body)
            return
            
        
        if self.body is not None:
            self._subdivide()
            
            old_body = self.body
            quadrant_idx_old = self._get_quadrant(old_body)
            self.children[quadrant_idx_old].insert(old_body)
            self.body = None 
            
            quadrant_idx_new = self._get_quadrant(body)
            self.children[quadrant_idx_new].insert(body)
            
            self.total_mass = old_body.mass
            self.center_of_mass = old_body.position.copy()
            self._update_mass(body)

    def _update_mass(self, body):
        """Updates the node's center of mass and total mass."""
        new_total_mass = self.total_mass + body.mass
        self.center_of_mass = (self.center_of_mass * self.total_mass + body.position * body.mass) / new_total_mass
        self.total_mass = new_total_mass