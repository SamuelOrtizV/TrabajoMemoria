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
                 start_up_multiplier=0.1,
                 penalty_low_rpms=-0.2,
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
                 hist_len=4,
                 time_step_duration=0.05,
                 buffer_lapse=0.5,
                 direction_threshold=0.2,
                 acc_x_threshold=0.1                             
                 ):
        """
        Instantiates a reward function for AC

        Args:
            reward_data_path: path where the trajectory file is stored
            max_mistakes: after this number of steps with no reward, episode is terminated
            steps_to_forget: number of steps to forget the previous mistakes
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            reward_checkpoint (float): Weight for reaching a checkpoint.
            reward_progress (float): Weight for progress on the track.
            reward_laps_weight (float): Weight for completing a lap.
            penalty_low_rpms (float): Penalty for low RPMs.
            penalty_backwards (float): Penalty for moving backwards.
            penalty_tyres_out (float): Penalty for going off track.
            penalty_car_damage (float): Penalty for car damage.
            penalty_collision (float): Penalty for continuous collision.
            threshold_speed (float): Minimum speed to receive speed reward.
            threshold_rpms (float): Minimum RPMs to avoid low RPM penalty.
            threshold_checkpoint (float): Threshold for track position to receive checkpoint reward.
            threshold_smooth_actions (float): Threshold for smooth actions.
            hist_len (int): Length of the history of images captured.
            time_step_duration (float): Duration of each time step in seconds.
            buffer_lapse (float): Duration of the collision detection buffer in seconds.
            direction_threshold (float): Threshold for steering direction detection.
            acc_x_threshold (float): Threshold for lateral acceleration detection in gs.
        """

      
        self.max_mistakes = max_mistakes
        self.steps_to_forget = steps_to_forget
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.reward_checkpoint = reward_checkpoint
        self.reward_progress = reward_progress
        self.reward_laps_weight = reward_laps_weight
        self.start_up_multiplier = start_up_multiplier
        self.penalty_low_rpms = penalty_low_rpms
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
        self.position_buffer = deque([0.0] * hist_len, maxlen=hist_len)
        self.buffer_size = int(buffer_lapse / time_step_duration) 
        self.steering_buffer = deque(maxlen=self.buffer_size)
        self.collision_buffer = deque(maxlen=self.buffer_size)
        self.direction_threshold = direction_threshold 
        self.acc_x_threshold = acc_x_threshold
        self.step_counter = 0
        self.mistake_counter = 0
        self.no_mistake_counter = 0

        self.previous_checkpoint = None
        self.previous_lap = 0

        self.car_damage = 0.0

        self.last_action = None  # last action taken by the car

        self.best = 0.0

    def compute_reward(self, telemetry_data, action):
        """
        Computes the current reward given the position pos
        Args:
            telemetry_data: Dictionary with the telemetry data of the car
            action: the action taken by the car (throttle-brake, steering)
        Returns:
            float, bool: the reward and the terminated signal
        """

        terminated = False        
        reward = 0.0
        self.step_counter += 1  # step counter to enable mistake counter
        mistake = False  # flag to check if a mistake happened
        collision = self.collision_detection(telemetry_data, action)

        track_position = 0.0 if telemetry_data["track_position"] >= 0.995 else telemetry_data["track_position"]
        self.position_buffer.append(track_position)  # we add the current position to the buffer
        progress = self.position_buffer[-1] - self.position_buffer[0]
        self.previous_checkpoint = track_position if self.previous_checkpoint is None else self.previous_checkpoint
        checkpoint_difference = track_position - self.previous_checkpoint

        # ----------------------REWARDS---------------------------

        # Reward for completing a lap
        if telemetry_data["laps"] > self.previous_lap:
            reward += self.reward_laps_weight
            self.previous_lap = telemetry_data["laps"]     

        # Reward for reaching a checkpoint
        if checkpoint_difference > self.threshold_checkpoint:
            reward += self.reward_checkpoint
            print(f"Checkpoint reached: {track_position}                                                                      ")
            self.previous_checkpoint = track_position

        # Reward for progress on the track
        if progress > 0.0:  # If we did progress on the track
            reward += self.reward_progress

        """ # Reward for speed
        if telemetry_data["speed"] > self.threshold_speed:
            reward += telemetry_data["speed"] / 400.0 """

        # ----------------------PENALTIES---------------------------

        """ # Penalty for non-smooth actions
        if self.last_action is not None:
            gas_brake_diff = abs(action[0] - self.last_action[0])
            wheel_diff = abs(action[1] - self.last_action[1])

            if gas_brake_diff > self.threshold_smooth_actions or wheel_diff > self.threshold_smooth_actions:
                reward += self.penalty_non_smooth_actions
            
        self.last_action = action  # we update the last action taken by the car """

        # Penalty for going off track
        """ if telemetry_data["tyres_out"] > 0:
            reward += self.penalty_tyres_out * telemetry_data["tyres_out"]
            mistake = True  """

        # Penalty and reward for acceleration when starting from 0
        if telemetry_data["gear"] == 1:
            reward += action[0] * self.start_up_multiplier

        # Penalty for low RPMs
        """ if telemetry_data["rpms"] < self.threshold_rpms*2:
            reward += self.penalty_low_rpms/2 """
        if telemetry_data["rpms"] < self.threshold_rpms:
            reward += self.penalty_low_rpms

        # Penalty for low speed or zero progress
        if telemetry_data["speed"] < self.threshold_speed or progress == 0: # and telemetry_data["gear"] > 1:  # If the car is not moving
            reward += self.penalty_low_speed
            mistake = True   

        # Penalty for moving backwards
        if progress < 0:
            reward += self.penalty_backwards
            mistake = True
        
        # Penalty for car damage
        if telemetry_data["car_damage"] > self.car_damage:            
            reward += self.penalty_car_damage #* (telemetry_data["car_damage"] - self.car_damage)
            self.car_damage = telemetry_data["car_damage"]

        # Penalty for continuous collision
        if collision and telemetry_data["car_damage"] > 0:
            reward += self.penalty_collision
            #mistake = True

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

        if telemetry_data["car_damage"] > 25:
            terminated = True  # The episode ends if the car is damaged

        max_reward = max(1, abs(reward))
        reward = reward / max_reward

        self.print_status(telemetry_data, action, collision)  # we print the status of the run

        return reward, terminated
    
    def collision_detection(self, telemetry_data, action):
        """
        Detects if the car is in a collision state
        """
        acc_x = telemetry_data["acc_x"]

        self.steering_buffer.append(action[1])  # Agrega la dirección al buffer de dirección

        avg_steering = sum(self.steering_buffer) / len(self.steering_buffer)

        # Evalúa el bloqueo lateral usando los promedios
        # Si se gira a la derecha y no hay aceleración lateral, o si se gira a la izquierda y no hay aceleración lateral, 
        # o si hay acelareción lateral y no hay giro en ese sentido, se considera que hay bloqueo lateral
        bloqueo_lateral = (avg_steering > self.direction_threshold and not (acc_x < -self.acc_x_threshold)) or \
                          (avg_steering < -self.direction_threshold and not (acc_x > self.acc_x_threshold)) or \
                          (acc_x > self.acc_x_threshold and avg_steering > 0.05) or \
                          (acc_x < -self.acc_x_threshold and avg_steering < -0.05)
        
        self.collision_buffer.append(bloqueo_lateral)

        avg_collision = sum(self.collision_buffer) / len(self.collision_buffer)

        # Si el auto no esta quieto y la mayoria de los frames del buffer son de bloqueo lateral, se considera que hay colisión
        if telemetry_data["speed"] > self.threshold_speed and avg_collision > 0.5:
            collision = True
        else:
            collision = False

        return collision

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        
        self.step_counter = 0
        self.mistake_counter = 0
        self.no_mistake_counter = 0
        
        self.previous_checkpoint = None
        self.previous_lap = 0

        self.car_damage = 0.0
        self.last_action = None

        self.position_buffer.clear()
        self.position_buffer.extend([0.0] * self.position_buffer.maxlen)

        self.steering_buffer.clear()
        self.collision_buffer.clear()

    def print_status(self, data, action, collision):
        """
        Prints the status of the run
        """

        if data["track_position"] > self.best and data["track_position"] < 0.995:
            self.best = data["track_position"]
        
        print(f"PR: {self.best} {data} Collision: {collision} Gas-Brake Turn: {np.round(action, 2)}                        ", end="\r")
        