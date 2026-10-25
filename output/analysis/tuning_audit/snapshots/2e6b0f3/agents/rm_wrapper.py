import gymnasium as gym

class HazardousDeliveryRMWrapper(gym.Wrapper):
    """
    Wraps the Hazardous Delivery environment to emit logical propositions
    needed for Reward Machines (QRM / PPO-RM). 
    The propositions are appended to the 'info' dictionary at each step.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # 1. Capture internal state before the step
        unwrapped = self.env.unwrapped
        pre_health = getattr(unwrapped, 'health', None)
        pre_carried = getattr(unwrapped, 'carried_packages', None)
        pre_delivered = getattr(unwrapped, 'delivered_packages', None)
        
        # 2. Execute the environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Determine logical propositions based on state changes
        propositions = set()
        
        if pre_health is not None:
            post_health = unwrapped.health
            post_carried = unwrapped.carried_packages
            post_delivered = unwrapped.delivered_packages
            
            # Damage and Health
            if post_health < pre_health:
                propositions.add("stepped_on_damage")
                if post_health <= 0:
                    propositions.add("died")
            if post_health > pre_health:
                propositions.add("healed")
                
            # Inventory
            if post_carried > pre_carried:
                propositions.add("picked_up_package")
            if post_delivered > pre_delivered:
                propositions.add("delivered_package")
                
            # Terminal Conditions
            if terminated:
                # Goal reached successfully
                if post_delivered >= unwrapped.num_packages and reward > 0:
                    propositions.add("reached_goal_successfully")
                # Goal reached without finishing deliveries
                elif reward == unwrapped.penalty_premature_goal:
                    propositions.add("reached_goal_prematurely")
                # Lava pit (instant death)
                elif reward == unwrapped.penalty_death_square and post_health > 0:
                    propositions.add("stepped_on_lava")
                    propositions.add("died")
                    
        # 4. Inject propositions into the info dictionary
        info["propositions"] = propositions
        
        return obs, reward, terminated, truncated, info

class DeferredMaintenanceRMWrapper(gym.Wrapper):
    """
    Wraps the Deferred Maintenance environment to emit logical propositions
    needed for Reward Machines (QRM / PPO-RM). 
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        unwrapped = self.env.unwrapped
        pre_score = unwrapped.score
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        diff = unwrapped.score - pre_score
        
        propositions = set()
        propositions.add(f"reward_{diff}")
        
        if unwrapped.readiness == 1.0:
            propositions.add("is_ready")
            
        info["propositions"] = propositions
        return obs, reward, terminated, truncated, info
