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
                 hist_len=4,
                 time_step_duration=0.05,
                 buffer_lapse=0.5,
                 direction_threshold=0.2,
                 acc_x_threshold=0.1,
                 mistake_collision=False,
                 mistale_out_of_track=False                             
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
            penalty_no_progress (float): Penalty for making no progress.
            penalty_backwards (float): Penalty for moving backwards.
            penalty_tyres_out (float): Penalty for going off track.
            penalty_car_damage (float): Penalty for car damage.
            penalty_collision (float): Penalty for continuous collision.
            threshold_speed (float): Minimum speed to receive speed reward.
            threshold_rpms (float): Minimum RPMs to avoid low RPM penalty.
            threshold_checkpoint (float): Threshold for track position to receive checkpoint reward.
            threshold_smooth_actions (float): Threshold for smooth actions.
            threshold_damage (float): Threshold for car damage to terminate the episode.
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

        self.previous_checkpoint = None
        self.previous_lap = 0

        self.car_damage = [0.0, 0.0, 0.0, 0.0]

        self.last_action = None  # last action taken by the car

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

        track_position = telemetry_data["track_position"]
        self.position_buffer.append(track_position)  # we add the current position to the buffer
        progress = self.position_buffer[-1] - self.position_buffer[0]
        if self.previous_checkpoint is None:
            self.previous_checkpoint = track_position if track_position < 0.95 else 0.0 # Some tracks start before pos 0.0
        checkpoint_difference = track_position - self.previous_checkpoint

        # ----------------------REWARDS---------------------------

        # Reward for completing a lap
        if telemetry_data["laps"] > self.previous_lap:
            #reward += self.reward_laps_weight
            #terminated = True
            #self.previous_lap = telemetry_data["laps"] 
            self.previous_checkpoint = 0.0

        # Reward for reaching a checkpoint
        if checkpoint_difference > self.threshold_checkpoint and checkpoint_difference < 0.5:
            reward += self.reward_checkpoint
            print(f"Checkpoint reached: {track_position}                                                                                                                                                              ")
            self.previous_checkpoint = track_position

        # Reward for progress on the track
        if progress > 0.0:  # If we did progress on the track
            reward += self.reward_progress * min((telemetry_data["speed"])/(self.threshold_speed*3), 1) #Full reward only if above speed threshold

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
        if telemetry_data["laps"] > self.previous_lap:
            reward += self.reward_laps_weight
            self.previous_lap = telemetry_data["laps"]
            terminated = True

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
                          (acc_x > self.acc_x_threshold and avg_steering > -0.05) or \
                          (acc_x < -self.acc_x_threshold and avg_steering < 0.05) # 0.05 is the steering deadzone for the analog controller anything between is considered 0
        
        self.collision_buffer.append(bloqueo_lateral)

        avg_collision = sum(self.collision_buffer) / len(self.collision_buffer)

        # Si el auto no esta quieto y la muchos de los frames del buffer son de bloqueo lateral, se considera que hay colisión
        if telemetry_data["speed"] > 1 and avg_collision > 0.5:
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

        self.car_damage = [0.0, 0.0, 0.0, 0.0]
        self.last_action = None

        self.position_buffer.clear()
        self.position_buffer.extend([0.0] * self.position_buffer.maxlen)

        self.steering_buffer.clear()
        self.collision_buffer.clear()

    def print_status(self, data, action, collision):
        """
        Prints the status of the run
        """      
        """ speed = data["speed"]
        rpms = data["rpms"] """

        print(f"{data} Collision: {collision} Gas-Brake Turn: {np.round(action, 2)}    ", end="\r")
        