
class RewardFunction:
    """
    Computes a reward from the telemetry data of the car.
    """
    def __init__(self,
                 max_mistakes=10,
                 steps_to_forget=100,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 reward_speed_weight=0.1,
                 reward_track_position_weight=5,
                 reward_laps_weight=500.0,
                 penalty_low_rpms=-0.2,
                 penalty_backwards=-0.5,
                 penalty_tyres_out=-0.5,
                 penalty_car_damage=-2.0,
                 threshold_speed=10.0,
                 threshold_rpms=2000.0,
                 threshold_checkpoint=0.001                              
                 ):
        """
        Instantiates a reward function for AC

        Args:
            reward_data_path: path where the trajectory file is stored
            max_mistakes: after this number of steps with no reward, episode is terminated
            steps_to_forget: number of steps to forget the previous mistakes
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            reward_speed_weight (float): Weight for speed reward.
            reward_track_position_weight (float): Weight for reaching a checkpoint.
            reward_laps_weight (float): Weight for completing a lap.
            penalty_low_rpms (float): Penalty for low RPMs.
            penalty_backwards (float): Penalty for moving backwards.
            penalty_tyres_out (float): Penalty for going off track.
            penalty_car_damage (float): Penalty for car damage.
            threshold_speed (float): Minimum speed to receive speed reward.
            threshold_rpms (float): Minimum RPMs to avoid low RPM penalty.
            threshold_checkpoint (float): Threshold for track position to receive checkpoint reward.
        """

      
        self.max_mistakes = max_mistakes
        self.steps_to_forget = steps_to_forget
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.reward_speed_weight = reward_speed_weight
        self.reward_track_position_weight = reward_track_position_weight
        self.reward_laps_weight = reward_laps_weight
        self.penalty_low_rpms = penalty_low_rpms
        self.penalty_backwards = penalty_backwards
        self.penalty_tyres_out = penalty_tyres_out
        self.penalty_car_damage = penalty_car_damage
        self.threshold_speed = threshold_speed
        self.threshold_rpms = threshold_rpms
        self.threshold_checkpoint = threshold_checkpoint

        self.step_counter = 0
        self.mistake_counter = 0
        self.no_mistake_counter = 0

        self.previous_checkpoint = 0.0
        self.previous_position = 0.0
        self.previous_lap = 0

    def compute_reward(self, telemetry_data):
        """
        Computes the current reward given the position pos
        Args:
            telemetry_data: Dictionary with the telemetry data of the car
        Returns:
            float, bool: the reward and the terminated signal
        """

        terminated = False        
        reward = 0.0
        self.step_counter += 1  # step counter to enable mistake counter
        mistake = False  # flag to check if a mistake happened

        # Reward for completing a lap
        if telemetry_data["laps"] > self.previous_lap:
            reward += self.reward_laps_weight
            self.previous_lap = telemetry_data["laps"]

        track_position = 0.0 if telemetry_data["track_position"] >= 0.995 else telemetry_data["track_position"]
        position_difference = track_position - self.previous_position
        checkpoint_difference = track_position - self.previous_checkpoint
        self.previous_position = track_position

        # Reward for minimum speed, this aims to keep the car moving
        if telemetry_data["speed"] >= self.threshold_speed:
            reward += self.reward_speed_weight
        else:
            reward -= self.reward_speed_weight


        # Reward for reaching a checkpoint
        if checkpoint_difference > self.threshold_checkpoint:  # If we did progress on the track
            reward += self.reward_track_position_weight
            print(f"Checkpoint reached: {track_position}")
            self.previous_checkpoint = track_position  # we update the previous checkpoint

        # Penalizations
        if telemetry_data["tyres_out"] > 0:
            reward += self.penalty_tyres_out * telemetry_data["tyres_out"]
            mistake = True 
        if telemetry_data["car_damage"] > 0:
            reward += self.penalty_car_damage
        if telemetry_data["rpms"] < self.threshold_rpms:
            reward += self.penalty_low_rpms
        if position_difference < 0:
            reward += self.penalty_backwards
            mistake = True

        # Termination condition
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

        terminated = terminated or telemetry_data["car_damage"] > 0  # The episode ends if the car is damaged

        return reward, terminated

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        
        self.step_counter = 0
        self.mistake_counter = 0
        self.no_mistake_counter = 0
        
        self.previous_checkpoint = 0.0
        self.previous_position = 0.0
        self.previous_lap = 0
