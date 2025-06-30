#!/usr/bin/env python3

import sympy as sp
import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, simplify, latex
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class DHMatrixGenerator:
    """
    Class to generate Denavit-Hartenberg transformation matrices
    for a 3-joint anthropomorphic arm
    """
    
    def __init__(self):
        # Define symbolic variables
        self.theta1, self.theta2, self.theta3 = symbols('theta_1 theta_2 theta_3')
        
        # DH Parameters for 3-joint anthropomorphic arm
        # Based on the notebook: r_1=0.0, r_2=1.0, r_3=1.0
        # Frame 0 is at same origin as Frame 1
        
        # Joint 1 (Base rotation about z-axis)
        self.dh_params = {
            1: {'theta': self.theta1, 'd': 0, 'a': 0, 'alpha': pi/2},
            2: {'theta': self.theta2, 'd': 0, 'a': 1, 'alpha': 0},
            3: {'theta': self.theta3, 'd': 0, 'a': 1, 'alpha': 0}
        }
        
        # Store matrices
        self.A0_1 = None
        self.A1_2 = None
        self.A2_3 = None
        self.A0_3 = None
        self.A0_3_simplified = None
        
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
    
    def generate_individual_matrices(self):
        """Generate individual transformation matrices A0_1, A1_2, A2_3"""
        
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
    
    def generate_composite_matrix(self):
        """Generate composite transformation matrix A0_3"""
        self.A0_3 = self.A0_1 * self.A1_2 * self.A2_3
        
        # Simplified version
        self.A0_3_simplified = simplify(self.A0_3)
    
    def save_matrix_as_png(self, matrix, filename):
        """
        Save a matrix as a PNG image

        Args:
            matrix: SymPy matrix to save
            filename: output filename
            title: title for the plot
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
                # Get element and format it
                element = str(matrix[i, j])
                element = self.format_mathematical_expression(element)
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
                fontsize=16, ha='center', va='center', fontfamily='monospace')

        # Save the figure
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}.png")
    
    def format_mathematical_expression(self, element_str):
        """
        Format a mathematical expression string with proper notation
        
        Args:
            element_str: String representation of matrix element
            
        Returns:
            Formatted string with proper mathematical notation
        """
        # Convert to string if not already
        element = str(element_str)
        
        # Replace theta variables with proper subscript notation
        # Be very explicit about the replacements
        element = element.replace('theta_1', 'θ₁')
        element = element.replace('theta_2', 'θ₂') 
        element = element.replace('theta_3', 'θ₃')
        
        # Also handle cases where SymPy might format differently
        element = element.replace('theta1', 'θ₁')
        element = element.replace('theta2', 'θ₂')
        element = element.replace('theta3', 'θ₃')
        
        # Clean up other mathematical notation
        element = element.replace('**', '^')
        element = element.replace('*', '')
        element = element.replace('pi', 'π')
        element = element.replace('Pi', 'π')
        
        return element

    def generate_all_matrices(self):
        """Generate all matrices and save as PNG images"""
        
        print("Generating DH transformation matrices...")
        
        # Generate individual matrices
        self.generate_individual_matrices()
        
        # Generate composite matrix
        self.generate_composite_matrix()
        
        # Save all matrices as PNG images
        self.save_matrix_as_png(self.A0_1, "A0_1")
        self.save_matrix_as_png(self.A1_2, "A1_2") 
        self.save_matrix_as_png(self.A2_3, "A2_3")
        self.save_matrix_as_png(self.A0_3, "A0_3")
        self.save_matrix_as_png(self.A0_3_simplified, "A0_3_simplified")
        
        print("All matrices generated successfully!")
        
    def get_matrices(self):
        """Return all computed matrices"""
        return {
            'A0_1': self.A0_1,
            'A1_2': self.A1_2, 
            'A2_3': self.A2_3,
            'A0_3': self.A0_3,
            'A0_3_simplified': self.A0_3_simplified
        }

def main():
    """Main function to generate matrices"""
    
    # Create generator instance
    generator = DHMatrixGenerator()
    
    # Generate all matrices and save as PNG
    generator.generate_all_matrices()
    
    # Print summary
    print("\nGenerated files:")
    files = ["A0_1.png", "A1_2.png", "A2_3.png", "A0_3.png", "A0_3_simplified.png"]
    for file in files:
        if os.path.exists(file):
            print(f"Success: {file}")
        else:
            print(f"Fail: {file}")

if __name__ == "__main__":
    main()