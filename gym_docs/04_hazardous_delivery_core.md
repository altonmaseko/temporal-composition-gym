# Hazardous Delivery Environment: Core Engine

The `HazardousDeliveryCoreEnv` defines the fundamental physics, rules, state transitions, and reward logic for the Hazardous Delivery task. Built on top of `MiniGridEnv`, it implements a gridworld logistics challenge where an agent must navigate hazards, manage its health, and deliver packages in a potentially strict sequential order.

All the core logic detailed below is located in:
> 📁 [`temporal_comp_gym/envs/hazardous_delivery/core.py`](../temporal_comp_gym/envs/hazardous_delivery/core.py)

---

## 1. Initialization and Configuration (`__init__`)

When the environment is instantiated, the `__init__` method sets up the constraints, rewards, and internal state trackers for the agent.

### Environment and Agent State Variables
The environment keeps track of the agent's dynamic state, such as health and package inventory, alongside the configurable hyperparameters.

```python
def __init__(self, grid_size=10, percent_death_squares=0.05, percent_damage_squares=0.05, ...):
    # ... parameter assignments ...
    
    # Internal agent state trackers
    self.max_health = 10
    self.health = self.max_health
    self.carried_packages = 0
    self.delivered_packages = 0
    
    # Set up the mission string and pass base variables to Minigrid
    mission_space = MissionSpace(mission_func=lambda: "Deliver all packages and reach the goal.")
    super().__init__(
        mission_space=mission_space,
        grid_size=grid_size,
        max_steps=max_steps,
        **kwargs
    )
```

### Action Space Override
Unlike default Minigrid environments where actions include turning left/right and moving forward relative to the agent's orientation, this environment overrides the action space to strict 2D directional movement.

```python
# Override action space to discrete directional: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
self.action_space = gym.spaces.Discrete(4)
```

---

## 2. Grid Generation and Reset Logic (`reset` & `_gen_grid`)

### The `reset` Method
The `reset` method is called at the start of every new episode. It cleanly resets the internal state trackers and defers to the parent class's reset logic, which in turn calls `_gen_grid` to populate the map.

```python
def reset(self, *, seed=None, options=None):
    self.health = self.max_health
    self.carried_packages = 0
    self.delivered_packages = 0
    return super().reset(seed=seed, options=options)
```

### Grid Dynamics (`_gen_grid`)
The grid generation logic places the walls, hazards, packages, destinations, and the agent. Notably, it handles assigning Sequence IDs (`seq_id`) to items if `task_dependencies` is enabled, locking the task into a rigid sequence.

```python
def _gen_grid(self, width, height):
    self.grid = Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    
    # Calculate hazard quantities based on grid size
    num_inner = (width - 2) * (height - 2)
    num_death = int(num_inner * self.percent_death_squares)
    # ...
    
    # Place Packages with optional sequence enforcement
    for i in range(self.num_packages):
        seq_id = (i + 1) if self.task_dependencies else None
        self.place_obj(Package(seq_id=seq_id))
        
    # Place Destinations with optional sequence enforcement
    for i in range(self.num_packages):
        seq_id = (i + 1) if self.task_dependencies else None
        self.place_obj(Destination(seq_id=seq_id))
        
    # ... (hazard and agent placement logic follows)
```

---

## 3. Processing Actions and State Transitions (`step`)

The `step` method is the heart of the engine. It takes an action, attempts to move the agent, triggers entity interactions, computes the reward, and returns the observation.

### Movement Logic
First, the action translates to an X,Y coordinate change. The agent only moves into the cell if it is completely empty or if the object occupying it explicitly allows overlapping.

```python
def step(self, action):
    # ... initialize reward and termination variables ...
    x, y = self.agent_pos
    if action == 0: y -= 1     # UP
    elif action == 1: y += 1   # DOWN
    elif action == 2: x -= 1   # LEFT
    elif action == 3: x += 1   # RIGHT
        
    fwd_cell = self.grid.get(x, y)
    
    # Only move if the cell is clear or the object on it is non-blocking
    if fwd_cell is None or fwd_cell.can_overlap():
        self.agent_pos = (x, y)
```

### Interaction, Rewards, and Penalties
After movement, the engine assesses the cell the agent is currently occupying and processes the interaction logic based on the object type.

**Hazards and Health:**
Stepping on a `damage` square depletes health. If health drops to zero, it behaves identically to a `lava` (death) square, terminating the episode immediately with a heavy penalty. `health` nodes fully restore the agent and are consumed in the process.

```python
curr_cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])

if curr_cell is not None:
    if curr_cell.type == 'lava':
        reward = self.penalty_death_square
        terminated = True
        
    elif curr_cell.type == 'damage':
        reward = self.penalty_damage
        self.health -= 1
        if self.health <= 0:
            terminated = True
            reward += self.penalty_death_square
            
    elif curr_cell.type == 'health':
        self.health = self.max_health
        # Consume the health node upon use
        self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
```

**Logistics (Packages and Destinations):**
The agent automatically picks up packages or drops them off at destinations if they have the inventory capacity and satisfy the temporal composition rules (i.e., collecting and delivering in exact `seq_id` order).

```python
    elif curr_cell.type == 'package':
        can_pickup = True
        if self.task_dependencies:
            # Enforce sequential order: must pick up the specific next package in line
            if curr_cell.seq_id != (self.carried_packages + self.delivered_packages + 1):
                can_pickup = False
                
        if can_pickup and self.carried_packages < self.carrying_capacity:
            self.carried_packages += 1
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None) # Consume package
            
    elif curr_cell.type == 'destination':
        can_deliver = True
        if self.task_dependencies:
            # Enforce sequential order: must deliver to the specific next destination
            if curr_cell.seq_id != (self.delivered_packages + 1):
                can_deliver = False
                
        if can_deliver and self.carried_packages > 0:
            self.carried_packages -= 1
            self.delivered_packages += 1
            reward = self.reward_delivery
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None) # Consume destination
```

**Goal Resolution:**
The goal square can only be entered safely if all deliveries are complete. If an agent steps on it too early, they fail the episode and trigger a massive penalty.

```python
    elif curr_cell.type == 'goal':
        if self.delivered_packages >= self.num_packages:
            reward = self.reward_task_completion
            terminated = True
        else:
            reward = self.penalty_premature_goal
            terminated = True
```

This strict combination of logical rules (health management, spatial navigation, and sequential task dependencies) is what drives the temporal composition challenge in this environment.
