#!/usr/bin/env python3

import rospy
import numpy as np
import math

class InverseKinematics:
    """
    Inverse Kinematics solver for 3-joint anthropomorphic arm
    """
    
    def __init__(self):
        """Initialize the inverse kinematics solver"""
        
        # DH Parameters - same as forward kinematics
        # Joint 1: theta1, d=0, a=0, alpha=π/2 (base rotation)
        # Joint 2: theta2, d=0, a=1, alpha=0 (shoulder)  
        # Joint 3: theta3, d=0, a=1, alpha=0 (elbow)
        
        self.r2 = 1.0  # Length of link 2
        self.r3 = 1.0  # Length of link 3
        
        # Joint limits (from notebook)
        self.theta2_min = -math.pi/4      # -pi/4
        self.theta2_max = 3*math.pi/4     # 3*pi/4
        self.theta3_min = -3*math.pi/4    # -3*pi/4  
        self.theta3_max = 3*math.pi/4     # 3*pi/4
        
        print("Inverse Kinematics solver initialized!")
        print(f"Joint limits:")
        print(f"  Theta 2: [{self.theta2_min:.6f}, {self.theta2_max:.6f}]")
        print(f"  Theta 3: [{self.theta3_min:.6f}, {self.theta3_max:.6f}]")
    
    def get_user_input(self):
        """
        Get target position from user input
        
        Returns:
            tuple: (x, y, z) target position
        """
        try:
            x = float(input("Enter the target x position: "))
            y = float(input("Enter the target y position: "))
            z = float(input("Enter the target z position: "))
            
            return x, y, z
            
        except ValueError:
            print("Error: Please enter valid numeric values!")
            return None, None, None
    
    def solve_inverse_kinematics(self, target_x, target_y, target_z):
        """
        Solve inverse kinematics for given target position
        
        Args:
            target_x, target_y, target_z: Target end effector position
            
        Returns:
            list: List of solution dictionaries with theta values and validity
        """
        
        solutions = []
        
        # Calculate theta1 from x,y position
        theta1_solutions = [
            math.atan2(target_y, target_x),  # Primary solution
            math.atan2(target_y, target_x) + math.pi  # Secondary solution (opposite direction)
        ]
        
        for theta1 in theta1_solutions:
            # Normalize theta1 to [-π, π]
            theta1 = math.atan2(math.sin(theta1), math.cos(theta1))
            
            # Calculate the projected distance in xy plane
            r_xy = math.sqrt(target_x**2 + target_y**2)
            
            # For each theta1, solve for theta2 and theta3
            # Using the constraint equations for the 3DOF arm
            
            # The end effector position in the plane defined by theta1 is:
            # r_projected = r2*cos(theta2) + r3*cos(theta2 + theta3)
            # z = r2*sin(theta2) + r3*sin(theta2 + theta3)
            
            # Distance from origin to target in the arm plane
            target_distance = math.sqrt(r_xy**2 + target_z**2)
            
            # Check if target is reachable
            if target_distance > (self.r2 + self.r3):
                print(f"Target too far for theta1={theta1:.6f}")
                continue
            if target_distance < abs(self.r2 - self.r3):
                print(f"Target too close for theta1={theta1:.6f}")
                continue
            
            # Solve for theta3 using law of cosines
            # cos(theta3) = (target_distance^2 - r2^2 - r3^2) / (2*r2*r3)
            cos_theta3 = (target_distance**2 - self.r2**2 - self.r3**2) / (2 * self.r2 * self.r3)
            
            # Check if solution exists
            if abs(cos_theta3) > 1.0:
                print(f"No solution for theta3 with theta1={theta1:.6f}")
                continue
            
            # Two solutions for theta3 (elbow up/down)
            sin_theta3_pos = math.sqrt(1 - cos_theta3**2)
            sin_theta3_neg = -sin_theta3_pos
            
            theta3_solutions = [
                math.atan2(sin_theta3_pos, cos_theta3),  # Elbow up
                math.atan2(sin_theta3_neg, cos_theta3)   # Elbow down
            ]
            
            for i, theta3 in enumerate(theta3_solutions):
                config = "plus" if i == 0 else "minus"
                
                # Calculate theta2
                # Using the constraint equations:
                # r_xy = r2*cos(theta2) + r3*cos(theta2 + theta3)
                # z = r2*sin(theta2) + r3*sin(theta2 + theta3)
                
                # Solve for theta2
                k1 = self.r2 + self.r3 * cos_theta3
                k2 = self.r3 * math.sin(theta3)
                
                # theta2 = atan2(z, r_xy) - atan2(k2, k1)
                theta2 = math.atan2(target_z, r_xy) - math.atan2(k2, k1)
                
                # Normalize angles to [-π, π]
                theta2 = math.atan2(math.sin(theta2), math.cos(theta2))
                theta3 = math.atan2(math.sin(theta3), math.cos(theta3))
                
                # Check joint limits
                theta2_valid = self.theta2_min <= theta2 <= self.theta2_max
                theta3_valid = self.theta3_min <= theta3 <= self.theta3_max
                solution_possible = theta2_valid and theta3_valid
                
                # Create solution dictionary
                solution = {
                    'theta1': theta1,
                    'theta2': theta2,
                    'theta3': theta3,
                    'theta2_config': 'plus' if theta1 >= 0 else 'minus',
                    'theta3_config': config,
                    'possible': solution_possible,
                    'theta2_valid': theta2_valid,
                    'theta3_valid': theta3_valid
                }
                
                solutions.append(solution)
        
        return solutions
    
    def print_solution_details(self, solution, target_x, target_y, target_z):
        """
        Print detailed information about a solution
        
        Args:
            solution: Solution dictionary
            target_x, target_y, target_z: Target position
        """
        print(f"Input Data===== theta_2_config CONFIG = {solution['theta2_config']}")
        print(f"Input Data===== theta_3_config CONFIG = {solution['theta3_config']}")
        print(f"Pee_x = {target_x}")
        print(f"Pee_y = {target_y}")
        print(f"Pee_z = {target_z}")
        print(f"r2 = {self.r2}")
        print(f"r3 = {self.r3}")
        
        # Calculate cos and sin values for verification
        cos_theta3 = math.cos(solution['theta3'])
        sin_theta3 = math.sin(solution['theta3'])
        cos_theta2 = math.cos(solution['theta2'])
        sin_theta2 = math.sin(solution['theta2'])
        cos_theta1 = math.cos(solution['theta1'])
        sin_theta1 = math.sin(solution['theta1'])
        
        print(f"theta_3 solution possible C3=={cos_theta3}, S3={sin_theta3}")
        print(f"theta_2 solution possible C2=={cos_theta2}, S2={sin_theta2}")
        print(f"theta_1 solution possible C1=={cos_theta1}, S1={sin_theta1}")
        
        angles = [solution['theta1'], solution['theta2'], solution['theta3']]
        print(f"Angles thetas solved ={angles}")
        
        # Check joint limits and print warnings
        if not solution['theta2_valid']:
            print(f">>>>>>>>>>>>>>> theta_2 NOT POSSIBLE, MIN={self.theta2_min}, theta_2={solution['theta2']}, MAX={self.theta2_max}")
        if not solution['theta3_valid']:
            print(f">>>>>>>>>>>>>>> theta_3 NOT POSSIBLE, MIN={self.theta3_min}, theta_3={solution['theta3']}, MAX={self.theta3_max}")
        
        print(f"possible_solution = {solution['possible']}")
    
    def display_results(self, solutions, target_x, target_y, target_z):
        """
        Display all solutions
        
        Args:
            solutions: List of solution dictionaries
            target_x, target_y, target_z: Target position
        """
        print(f"\n" + "="*60)
        print("INVERSE KINEMATICS RESULTS")
        print("="*60)
        print(f"Target position: [{target_x}, {target_y}, {target_z}]")
        print(f"Found {len(solutions)} solution(s):")
        print("="*60)
        
        for i, solution in enumerate(solutions):
            print(f"\n--- Solution {i+1} ---")
            self.print_solution_details(solution, target_x, target_y, target_z)
        
        print(f"\n" + "="*60)
        print("SUMMARY:")
        for i, solution in enumerate(solutions):
            angles = [solution['theta1'], solution['theta2'], solution['theta3']]
            print(f"Angles thetas solved ={angles} , solution possible = {solution['possible']}")
        print("="*60)
    
    def run(self):
        """
        Main execution function
        """
        print("Inverse Kinematics for 3-joint Anthropomorphic Arm")
        print("="*55)
        
        # Get user input
        target_x, target_y, target_z = self.get_user_input()
        
        if target_x is None:
            print("Invalid input. Exiting...")
            return
        
        # Solve inverse kinematics
        try:
            solutions = self.solve_inverse_kinematics(target_x, target_y, target_z)
            
            if not solutions:
                print("No valid solutions found!")
                return
            
            # Display results
            self.display_results(solutions, target_x, target_y, target_z)
            
            print(f"\nInverse kinematics computation completed successfully!")
            
        except Exception as e:
            print(f"Error during computation: {e}")
            return

def main():
    """Main function"""
    
    # Initialize ROS node (optional, for ROS integration)
    try:
        rospy.init_node('ik_antropomorphic_arm', anonymous=True)
    except:
        pass  # Continue without ROS if not available
    
    # Create and run inverse kinematics solver
    ik_solver = InverseKinematics()
    ik_solver.run()

if __name__ == "__main__":
    main()