import numpy as np
from collections import deque

class RewardFunction:
    """
    Computes a reward from the telemetry data of the car.
    """
    def __init__(self,
                 max_mistakes=10,
                 steps_to_forget=100,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 reward_checkpoint=5,
                 reward_progress=0.1,
                 reward_laps_weight=500.0,
                 penalty_no_progress=-0.2,
                 penalty_low_speed=-0.2,
                 penalty_backwards=-0.5,
                 penalty_tyres_out=-0.5,
                 penalty_car_damage=-2.0,
                 penalty_collision=-0.5,
                 penalty_non_smooth_actions=-0.01,
                 threshold_speed=10.0,
                 threshold_rpms=2000.0,
                 threshold_checkpoint=0.01,
                 threshold_smooth_actions=0.1,
                 threshold_damage=25.0,
                 max_speed=120.0,
                 hist_len=4,
                 time_step_duration=0.05,
                 buffer_lapse=0.5,
                 direction_threshold=0.2,
                 acc_x_threshold=0.1,
                 mistake_collision=False,
                 mistale_out_of_track=False                             
                 ):
        """Instantiate reward function for AC with configurable weights and thresholds.

        Args:
            max_mistakes: after this number of steps with mistakes, episode terminates
            steps_to_forget: steps without mistakes before resetting counters
            min_nb_steps_before_failure: minimum steps before allowing termination
            reward_checkpoint: reward for passing checkpoints
            reward_progress: reward weight for forward progress
            reward_laps_weight: reward for completing a lap
            penalty_no_progress: penalty for no progress
            penalty_low_speed: penalty below speed threshold
            penalty_backwards: penalty for negative progress
            penalty_tyres_out: penalty for wheels off track
            penalty_car_damage: penalty for increased damage
            penalty_collision: penalty when collision detected
            penalty_non_smooth_actions: penalty for jerky steering (disabled in code)
            threshold_speed: speed threshold for speed-related reward/penalty
            threshold_rpms: RPM threshold to avoid low-RPM penalty (unused)
            threshold_checkpoint: delta threshold to count a new checkpoint
            threshold_smooth_actions: steering delta considered jerky
            threshold_damage: immediate termination when damage exceeds this
            max_speed: speed cap used in some reward calculations
            hist_len: history length for position buffer
            time_step_duration: duration of each environment step (seconds)
            buffer_lapse: time window for collision detection buffers (seconds)
            direction_threshold: steering threshold for lock/understeer detection
            acc_x_threshold: lateral acceleration threshold for lock detection
        """

      
        self.max_mistakes = max_mistakes
        self.steps_to_forget = steps_to_forget
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.reward_checkpoint = reward_checkpoint
        self.reward_progress = reward_progress
        self.reward_laps_weight = reward_laps_weight
        self.penalty_no_progress = penalty_no_progress
        self.penalty_low_speed = penalty_low_speed
        self.penalty_backwards = penalty_backwards
        self.penalty_tyres_out = penalty_tyres_out
        self.penalty_car_damage = penalty_car_damage
        self.penalty_collision = penalty_collision
        self.penalty_non_smooth_actions = penalty_non_smooth_actions
        self.threshold_speed = threshold_speed
        self.threshold_rpms = threshold_rpms
        self.threshold_checkpoint = threshold_checkpoint
        self.threshold_smooth_actions = threshold_smooth_actions
        self.threshold_damage = threshold_damage
        self.max_speed = max_speed
        self.position_buffer = deque([0.0] * hist_len, maxlen=hist_len)
        self.buffer_size = int(buffer_lapse / time_step_duration) 
        self.steering_buffer = deque(maxlen=self.buffer_size)
        self.collision_buffer = deque(maxlen=self.buffer_size)
        self.direction_threshold = direction_threshold 
        self.acc_x_threshold = acc_x_threshold
        self.mistake_collision = mistake_collision
        self.mistake_out_of_track = mistale_out_of_track

        self.step_counter = 0
        self.mistake_counter = 0
        self.no_mistake_counter = 0

        self.track_position = 0.0
        self.previous_checkpoint = 0
        self.previous_lap = 0

        self.car_damage = [0.0, 0.0, 0.0, 0.0]

        self.start_position = None
        self.prev_pos = 0.0
        self.lap_completed = False  # flag to check if a lap was completed

        self.last_action = None  # last action taken by the car

    def compute_reward(self, telemetry_data, action):
        """Compute reward and termination from telemetry and action.

        Args:
            telemetry_data: dict containing the required keys
            action: array-like [throttle_brake, steering]
        Returns:
            (reward: float, terminated: bool)
        """

        terminated = False        
        reward = 0.0
        self.step_counter += 1  # step counter to enable mistake counter
        mistake = False  # flag to check if a mistake happened
        self.steering_buffer.append(action[1])  # Agrega la dirección al buffer de dirección
        collision = self.collision_detection(telemetry_data, action)

        og_track_position = telemetry_data["track_position"]        

        if self.start_position is None:
            self.start_position = og_track_position # we set the start position of the car

        if abs(self.start_position - og_track_position)> 0.0001:  #Avoids issues when the car goes a bit back at the start
            self.track_position = round((og_track_position - self.start_position) % 1.0, 5) # we normalize the track position to [0, 1]

        #print(f"Track position: {self.track_position}, Start position: {self.start_position}, OG Track position: {og_track_position}")

        self.position_buffer.append(self.track_position)  # we add the current position to the buffer
        progress = self.position_buffer[-1] - self.position_buffer[0]
            
        checkpoint_difference = self.track_position - self.previous_checkpoint

        if self.track_position < 0.2 and self.prev_pos > 0.8:
            self.lap_completed = True  # El auto acaba de dar una vuelta

        self.prev_pos = self.track_position

        # ----------------------REWARDS---------------------------

        # Reward for completing a lap
        """ if self.lap_completed: #telemetry_data["laps"] > self.previous_lap:
            #reward += self.reward_laps_weight
            #terminated = True
            #self.previous_lap = telemetry_data["laps"] 
            self.previous_checkpoint = 0.0
        """
        # Reward for reaching a checkpoint
        if checkpoint_difference > self.threshold_checkpoint and checkpoint_difference < 0.5:
            reward += self.reward_checkpoint
            #print(f"Checkpoint reached: {track_position}                                                                                                                                                              ")
            self.previous_checkpoint = self.track_position

        # Reward for progress on the track
        if progress > 0.0:  # If we did progress on the track
            epislon = 0.0001  # Small value to avoid division by zero
            reward += self.reward_progress * min((telemetry_data["speed"])/(self.max_speed+epislon), 1) #Full reward only if driving at max speed allowed

        """ # Reward for speed
        if telemetry_data["speed"] > self.threshold_speed:
            reward += telemetry_data["speed"] / 400.0 """

        # ----------------------PENALTIES---------------------------

        """ # Penalty for non-smooth actions #TODO: implementar algo que compare la ultima accion con el buffer de giros
        if self.last_action is not None:
            #gas_brake_diff = abs(action[0] - self.last_action[0])
            wheel_diff = abs(action[1] - self.last_action[1])

            if wheel_diff > self.threshold_smooth_actions:
                reward += self.penalty_non_smooth_actions
            
        self.last_action = action  # we update the last action taken by the car """

        # Penalty for going off track
        if telemetry_data["tyres_out"] > 0:
            reward += self.penalty_tyres_out * telemetry_data["tyres_out"]
            if self.mistake_out_of_track:
                mistake = True 

        # Penalty for low speed
        if telemetry_data["speed"] < self.threshold_speed:
            # Penalization decreases linearly from 0 to penalty_low_speed as speed decreases from threshold_speed to 0
            reward += self.penalty_low_speed*(self.threshold_speed - telemetry_data["speed"])/self.threshold_speed
            #reward += action[0] * self.start_up_multiplier
            #mistake = True
        
        # Penalty for zero progress
        if progress == 0:
            reward += self.penalty_no_progress
            mistake = True

        # Penalty for moving backwards
        if progress < 0:
            reward += self.penalty_backwards
            mistake = True
        
        # Penalty for car damage
        for i in range(len(telemetry_data["car_damage"])):
            if telemetry_data["car_damage"][i] > self.car_damage[i]:            
                reward += self.penalty_car_damage #* (telemetry_data["car_damage"] - self.car_damage)
                self.car_damage[i] = telemetry_data["car_damage"][i]

        # Penalty for continuous collision
        if collision and max(telemetry_data["car_damage"]) > 0:
            reward += self.penalty_collision
            if self.mistake_collision:
                mistake = True

        # Termination conditions
        if mistake:
            # If mistake happens for too many steps, the episode terminates
            self.no_mistake_counter = 0  # we reset the no mistake counter
            if self.step_counter > self.min_nb_steps_before_failure:
                self.mistake_counter += 1
                if self.mistake_counter > self.max_mistakes:
                    terminated = True
        else:
            self.no_mistake_counter += 1  # we count the number of steps without mistakes
            if self.no_mistake_counter > self.steps_to_forget:
                self.mistake_counter = 0
                self.no_mistake_counter = 0

        if max(telemetry_data["car_damage"]) > self.threshold_damage:
            terminated = True  # The episode ends if the car is damaged beyond the threshold

        reward = np.clip(reward, -1.0, 1.0)  # Clip the reward to [-1, 1]

        # Reward for completing a lap
        if self.lap_completed: #telemetry_data["laps"] > self.previous_lap:
            print(f"\n---Lap completed---\n")  # Print the lap number
            reward += self.reward_laps_weight
            self.previous_lap = telemetry_data["laps"]
            self.track_position = 1.0
            terminated = True

        self.print_status(telemetry_data, action, collision, reward)  # we print the status of the run

        return reward, terminated
    
    def collision_detection(self, telemetry_data, action):
        """
        Detects if the car is in a collision state
        """
        acc_x = telemetry_data["acc_x"]

        avg_steering = sum(self.steering_buffer) / len(self.steering_buffer)

    # Lateral lock detection using averages:
    # - turning right but no leftward acceleration
    # - turning left but no rightward acceleration
    # - lateral acceleration without steering in that direction
        bloqueo_lateral = (avg_steering > self.direction_threshold and not (acc_x < -self.acc_x_threshold)) or \
                          (avg_steering < -self.direction_threshold and not (acc_x > self.acc_x_threshold)) or \
                          (acc_x > self.acc_x_threshold and avg_steering > -0.05) or \
                          (acc_x < -self.acc_x_threshold and avg_steering < 0.05) # 0.05 is the steering deadzone for the analog controller anything between is considered 0
        
        self.collision_buffer.append(bloqueo_lateral)

        avg_collision = sum(self.collision_buffer) / len(self.collision_buffer)

    # If the car is moving and most frames indicate lateral lock, consider collision
        if telemetry_data["speed"] > 1 and avg_collision > 0.5:
            collision = True
        else:
            collision = False

        return collision

    def reset(self):
        """Reset internal state for a new episode."""
        
        self.step_counter = 0
        self.mistake_counter = 0
        self.no_mistake_counter = 0
        
        self.previous_checkpoint = 0
        self.previous_lap = 0

        self.car_damage = [0.0, 0.0, 0.0, 0.0]
        self.last_action = None

        self.start_position = None
        self.prev_pos = 0.0
        self.track_position = 0.0
        self.lap_completed = False

        self.position_buffer.clear()
        self.position_buffer.extend([0.0] * self.position_buffer.maxlen)

        self.steering_buffer.clear()
        self.collision_buffer.clear()

    def print_status(self, data, action, collision, reward):
        """Print the current status (debug only)."""

        print(f"Speed: {data['speed']:.2f}, RPMS: {data['rpms']:.2f}, Gear: {data['gear']}, Reward: {reward:.4f}, Track Position: {data['track_position']}, Progress: {self.track_position}, Damage: {np.round(max(data['car_damage']), 2)}, Collision: {collision}, Gas-Brake Turn: {np.round(action, 2)}    ", end="\r")
        