import numpy as np
from physics_sim import PhysicsSim

class LandingTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.sim_init_pose = self.sim.pose # saving initial position of the quadcopter

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Taking into account the deviation between the current position and the target position on the xy axis
        reward = (self.target_pos[2] - self.sim.pose[2]) - 0.5 * (abs(self.target_pos - self.sim.pose[:3]))[:2].sum()
        
        
        if self.sim.pose[2] <= self.target_pos[2]:
            reward += 140.0  # Giving positive point for passing the targeted Z axis
        else:
            reward -= 10     # Punishing for each step if the agent is still above the ground
            
        # Giving agent positive reward if at each step if the current position is 
        reward += (self.sim_init_pose - self.sim.pose)[2]
        
        # Bringing the whole reward in the range of [-1, 1] and summing it to give to give the range of [-3, -3] here in this case
        reward = np.tanh(reward).sum()
        
        
        
        
        # My last reward function
#         reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, self.sim.pose[0], self.sim.pose[1], self.sim.pose[2], done # Also passing the final value of x, y, z

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state