#!/usr/bin/env python3

import rospy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, pi, Matrix, simplify, N

class ForwardKinematics:
    """
    Forward Kinematics solver for 3-joint anthropomorphic arm
    """
    
    def __init__(self):
        """Initialize the forward kinematics solver"""
        
        # Define symbolic variables
        self.theta1, self.theta2, self.theta3 = symbols('theta_1 theta_2 theta_3')
        
        # DH Parameters for 3-joint anthropomorphic arm
        # Based on the notebook: r_1=0.0, r_2=1.0, r_3=1.0
        self.dh_params = {
            1: {'theta': self.theta1, 'd': 0, 'a': 0, 'alpha': pi/2},
            2: {'theta': self.theta2, 'd': 0, 'a': 1, 'alpha': 0},
            3: {'theta': self.theta3, 'd': 0, 'a': 1, 'alpha': 0}
        }
        
        # Generate the symbolic matrices
        self.generate_matrices()
        
        print("Forward Kinematics solver initialized!")
        print("DH matrices generated successfully.")
    
    def create_dh_matrix(self, theta, d, a, alpha):
        """
        Create a single DH transformation matrix
        
        Args:
            theta: joint angle
            d: link offset
            a: link length  
            alpha: link twist
            
        Returns:
            4x4 transformation matrix
        """
        return Matrix([
            [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0,           sin(alpha),             cos(alpha),             d],
            [0,           0,                      0,                      1]
        ])
    
    def generate_matrices(self):
        """Generate DH transformation matrices"""
        
        # A0_1 matrix
        params_1 = self.dh_params[1]
        self.A0_1 = self.create_dh_matrix(
            params_1['theta'], params_1['d'], 
            params_1['a'], params_1['alpha']
        )
        
        # A1_2 matrix  
        params_2 = self.dh_params[2]
        self.A1_2 = self.create_dh_matrix(
            params_2['theta'], params_2['d'],
            params_2['a'], params_2['alpha']
        )
        
        # A2_3 matrix
        params_3 = self.dh_params[3]
        self.A2_3 = self.create_dh_matrix(
            params_3['theta'], params_3['d'],
            params_3['a'], params_3['alpha']
        )
        
        # Composite matrix
        self.A0_3 = self.A0_1 * self.A1_2 * self.A2_3
        
        # Simplified version
        self.A0_3_simplified = simplify(self.A0_3)
    
    def get_user_input(self):
        """
        Get theta values from user input
        
        Returns:
            tuple: (theta1, theta2, theta3) values in radians
        """
        try:
            theta1 = float(input("Enter the value for theta_1: "))
            theta2 = float(input("Enter the value for theta_2: "))
            theta3 = float(input("Enter the value for theta_3: "))
            
            return theta1, theta2, theta3
            
        except ValueError:
            print("Error: Please enter valid numeric values!")
            return None, None, None
    
    def compute_forward_kinematics(self, theta1_val, theta2_val, theta3_val):
        """
        Compute forward kinematics for given theta values
        
        Args:
            theta1_val: Value for theta1 in radians
            theta2_val: Value for theta2 in radians  
            theta3_val: Value for theta3 in radians
            
        Returns:
            tuple: (position_matrix, orientation_matrix, full_matrix)
        """
        
        # Substitute the actual theta values into the simplified matrix
        A0_3_evaluated = self.A0_3_simplified.subs([
            (self.theta1, theta1_val),
            (self.theta2, theta2_val), 
            (self.theta3, theta3_val)
        ])
        
        # Convert to numerical values and round for cleaner output
        A0_3_numerical = Matrix([[self.clean_number(N(A0_3_evaluated[i, j], 15)) for j in range(4)] 
                                for i in range(4)])
        
        # Extract position (last column, first 3 rows)
        position_matrix = Matrix([
            [A0_3_numerical[0, 3]],
            [A0_3_numerical[1, 3]], 
            [A0_3_numerical[2, 3]]
        ])
        
        # Extract orientation (first 3x3 submatrix)
        orientation_matrix = Matrix([
            [A0_3_numerical[0, 0], A0_3_numerical[0, 1], A0_3_numerical[0, 2]],
            [A0_3_numerical[1, 0], A0_3_numerical[1, 1], A0_3_numerical[1, 2]],
            [A0_3_numerical[2, 0], A0_3_numerical[2, 1], A0_3_numerical[2, 2]]
        ])
        
        return position_matrix, orientation_matrix, A0_3_numerical
    
    def clean_number(self, num):
        """
        Clean up numerical values for better display
        
        Args:
            num: Numerical value to clean
            
        Returns:
            Cleaned number with appropriate precision
        """
        # Convert to float first
        val = float(num)
        
        # Round to 15 decimal places to remove floating point errors
        val = round(val, 15)
        
        # If the value is very close to zero, make it exactly zero
        if abs(val) < 1e-10:
            return 0
        
        # Check if it's close to an integer
        if abs(val - round(val)) < 1e-10:
            return int(round(val))
        
        # If the value is very close to a simple fraction, round it
        # Check for common fractions like 0.5, -0.5, etc.
        rounded_val = round(val, 1)
        if abs(val - rounded_val) < 1e-10:
            return rounded_val
        
        # Otherwise, return with 12 decimal places but strip trailing zeros
        return round(val, 15)
    
    def save_evaluated_matrix(self, matrix, theta1_val, theta2_val, theta3_val):
        """
        Save the evaluated matrix as PNG image
        
        Args:
            matrix: Evaluated transformation matrix
            theta1_val, theta2_val, theta3_val: The theta values used
        """
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.axis('off')
        
        # Get matrix dimensions
        rows, cols = matrix.shape
        
        # First pass: format all elements and find maximum width for each column
        formatted_elements = []
        col_widths = [0] * cols
        
        for i in range(rows):
            row_elements = []
            for j in range(cols):
                # Get element and format it cleanly
                element = str(self.clean_number(float(matrix[i, j])))
                row_elements.append(element)
                # Track maximum width for this column
                col_widths[j] = max(col_widths[j], len(element))
            formatted_elements.append(row_elements)
        
        # Add padding to column widths for better spacing
        col_widths = [w + 2 for w in col_widths]
        
        # Create formatted matrix with proper bracket positioning
        formatted_matrix = ""
        
        for i in range(rows):
            # Choose proper connected bracket characters
            if rows == 1:
                left_bracket = "["
                right_bracket = "]"
            else:
                if i == 0:
                    left_bracket = "┌"
                    right_bracket = "┐"
                elif i == rows - 1:
                    left_bracket = "└"
                    right_bracket = "┘"
                else:
                    left_bracket = "│"
                    right_bracket = "│"
            
            # Build the content part (center-aligned columns)
            content_parts = []
            for j in range(cols):
                element = formatted_elements[i][j]
                # Center align each element within its column width
                centered_element = f"{element:^{col_widths[j]}}"
                content_parts.append(centered_element)
            
            # Join all columns with spaces between them
            content = "  ".join(content_parts)
            
            # Build final row: left_bracket + space + content + space + right_bracket
            row_str = left_bracket + " " + content + " " + right_bracket
            formatted_matrix += row_str + "\n"
        
        # Display matrix using monospace font for perfect alignment
        ax.text(0.5, 0.5, formatted_matrix, transform=ax.transAxes,
                fontsize=18, ha='center', va='center', fontfamily='monospace')
        
        # Save the figure
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        plt.savefig('A03_simplify_evaluated.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved A03_simplify_evaluated.png")
    
    def display_results(self, position, orientation, theta1_val, theta2_val, theta3_val):
        """
        Display the forward kinematics results
        
        Args:
            position: Position matrix (3x1)
            orientation: Orientation matrix (3x3)
            theta1_val, theta2_val, theta3_val: Input theta values
        """
        
        print(f"\n" + "="*50)
        print("FORWARD KINEMATICS RESULTS")
        print("="*50)
        print(f"Input angles:")
        print(f"  theta_1 = {theta1_val}")
        print(f"  theta_2 = {theta2_val}")
        print(f"  theta_3 = {theta3_val}")
        print(f"\nPosition Matrix:")
        print(position)
        print(f"\nOrientation Matrix:")
        print(orientation)
        print("="*50)
    
    def run(self):
        """
        Main execution function
        """
        print("Forward Kinematics for 3-joint Anthropomorphic Arm")
        print("="*55)
        
        # Get user input
        theta1_val, theta2_val, theta3_val = self.get_user_input()
        
        if theta1_val is None:
            print("Invalid input. Exiting...")
            return
        
        # Compute forward kinematics
        try:
            position, orientation, full_matrix = self.compute_forward_kinematics(
                theta1_val, theta2_val, theta3_val
            )
            
            # Display results
            self.display_results(position, orientation, theta1_val, theta2_val, theta3_val)
            
            # Save evaluated matrix image
            self.save_evaluated_matrix(full_matrix, theta1_val, theta2_val, theta3_val)
            
            print(f"\nForward kinematics computation completed successfully!")
            
        except Exception as e:
            print(f"Error during computation: {e}")
            return

def main():
    """Main function"""
    
    # Initialize ROS node
    try:
        rospy.init_node('fk_antropomorphic_arm', anonymous=True)
    except:
        # Continue without ROS if not available
        pass
    
    # Create and run forward kinematics solver
    fk_solver = ForwardKinematics()
    fk_solver.run()

if __name__ == "__main__":
    main()