# utils/expert_coach.py
import numpy as np
import random

class ExpertCoach:
    def __init__(self, exercise_type="bicep_curl"):
        self.exercise_type = exercise_type
        self.rep_history = []
        self.last_tip_time = 0
        self.tip_interval = 10  # Show tip every 10 seconds

        # Angle guidelines for bicep curl
        self.angle_guidelines = {
            "optimal_range": (60, 150),
            "full_contraction": 60,
            "ideal_midpoint": 90,
            "starting_position": 180,
            "partial_rep_threshold": 70,
            "elbow_swing_threshold": 25
        }

        # Coaching tips database
        self.tips = {
            "range_of_motion": [
                "Aim for {target}° - that's your optimal contraction point!",
                "Try to reach {full_contraction}° at the top for full bicep engagement",
                "Your current range is {current}°, ideal is {optimal_low}°-{optimal_high}°"
            ],
            "form": [
                "Keep elbows glued to your sides! Current swing: {swing}°",
                "Don't let your shoulders shrug up",
                "Keep your back straight and core tight"
            ],
            "tempo": [
                "Control the weight on the way down - count 2-3 seconds",
                "Explode up, control down - that's the perfect tempo!",
                "Avoid using momentum - make your biceps do the work"
            ],
            "general": [
                "Squeeze at the top for 1 second for maximum contraction",
                "Breathe out when lifting, in when lowering",
                "Focus on mind-muscle connection - visualize your biceps working"
            ]
        }

    def analyze_angle(self, elbow_angle, torso_angle=0):
        """Analyze current angle and provide feedback"""
        feedback = {
            "angle_analysis": "",
            "suggestion": "",
            "is_optimal": False,
            "target_angle": self.angle_guidelines["ideal_midpoint"]
        }

        # Check range of motion
        if elbow_angle < self.angle_guidelines["optimal_range"][0]:
            feedback["angle_analysis"] = f"Over-flexed: {int(elbow_angle)}°"
            feedback["suggestion"] = "Stop 3/4 of the way up to maintain tension"
            feedback["target_angle"] = self.angle_guidelines["optimal_range"][0]

        elif elbow_angle > self.angle_guidelines["optimal_range"][1]:
            feedback["angle_analysis"] = f"Too extended: {int(elbow_angle)}°"
            feedback["suggestion"] = f"Bend to {self.angle_guidelines['optimal_range'][1]}° for better bicep activation"
            feedback["target_angle"] = self.angle_guidelines["optimal_range"][1]

        elif self.angle_guidelines["optimal_range"][0] <= elbow_angle <= self.angle_guidelines["optimal_range"][1]:
            feedback["angle_analysis"] = f"Perfect range! {int(elbow_angle)}°"
            feedback["suggestion"] = "Excellent form - keep it up!"
            feedback["is_optimal"] = True

        else:
            feedback["angle_analysis"] = f"Current: {int(elbow_angle)}°"
            feedback["suggestion"] = f"Aim for {self.angle_guidelines['ideal_midpoint']}° for optimal tension"

        # Check for elbow swing
        if torso_angle > self.angle_guidelines["elbow_swing_threshold"]:
            feedback["angle_analysis"] += f" | Elbow swing: {int(torso_angle)}°"
            feedback["suggestion"] = "Press elbows against your sides to isolate biceps"

        return feedback

    def get_coaching_tip(self, elbow_angle=0, torso_angle=0):
        """Get a random coaching tip based on current form"""
        import time

        current_time = time.time()

        # Only show tip every X seconds
        if current_time - self.last_tip_time < self.tip_interval:
            return None

        self.last_tip_time = current_time

        # Choose tip category based on current form
        if elbow_angle < 60 or elbow_angle > 150:
            category = "range_of_motion"
        elif torso_angle > 20:
            category = "form"
        else:
            # Randomly choose between tempo and general
            category = random.choice(["tempo", "general"])

        # Get random tip from category
        tip_template = random.choice(self.tips[category])

        # Format tip with current values
        tip = tip_template.format(
            target=self.angle_guidelines["full_contraction"],
            full_contraction=self.angle_guidelines["full_contraction"],
            current=elbow_angle,
            optimal_low=self.angle_guidelines["optimal_range"][0],
            optimal_high=self.angle_guidelines["optimal_range"][1],
            swing=torso_angle
        )

        return tip

    def get_exercise_guidelines(self):
        """Get general guidelines for the exercise"""
        return {
            "exercise": self.exercise_type,
            "optimal_range": self.angle_guidelines["optimal_range"],
            "full_contraction": self.angle_guidelines["full_contraction"],
            "starting_position": self.angle_guidelines["starting_position"],
            "tips": [
                f"Start at {self.angle_guidelines['starting_position']}° (arm straight)",
                f"Contract to {self.angle_guidelines['full_contraction']}° at the top",
                f"Maintain {self.angle_guidelines['optimal_range'][0]}°-{self.angle_guidelines['optimal_range'][1]}° during movement",
                "Keep elbow swing under 25°"
            ]
        }
