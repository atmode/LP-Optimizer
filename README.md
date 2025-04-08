## LP-Optimizer

LP-Optimizer is a Python implementation for solving linear programming problems. This code allows users to define objective functions and constraints in a simple string format, which are then parsed and processed to find optimal solutions.

### Key Features

- **Objective Function Parsing**: Supports both maximization and minimization of linear functions.
- **Constraint Handling**: Easily input multiple constraints using standard inequality operators.
- **Feasibility Check**: Determines if a solution exists within the defined constraints.
- **Optimal Solution Calculation**: Computes the optimal values and corresponding solutions based on the defined objective.

### Usage

To use the LP-Optimizer, define your objective function and constraints as strings, and call the `optimize` function. The code will output the optimal solutions and values, along with details about feasible directions and corner points.

### Example

```python
obj1 = "max z = 2x + 3y"
cond1 = """
x + y <= 5
x <= 3
y <= 4
x >= 0
y >= 0
"""

optimize(obj1, cond1)
```
