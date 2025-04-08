from itertools import combinations
import re
import numpy as np

def parse_objective(obj_func):
    """
    Parses the objective function to identify its type and coefficients.
        obj_func (str): Objective function, e.g., "max z = 2x + 3y".
    """
    match = re.match(r"(min|max)\s+z\s*=\s*(.+)", obj_func.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Invalid objective function format. Example: 'min z = 2x + 3x_1 + 43y'")

    opt_type = match.group(1).lower()
    coefficients = extract_coefficients(match.group(2))
    return opt_type, coefficients

def parse_constraint(inequal):
    """
    Parses a constraint to extract coefficients, operator, and constant.
        inequal (str): Constraint in string format, e.g., "x + y <= 5".
    """
    operators = ["<=", ">=", "<", ">"]
    operator = next((op for op in operators if op in inequal), None)
    if not operator:
        raise ValueError("Invalid constraint format. Allowed operators: <=, >=, <, >")

    lhs, constant = inequal.split(operator)
    coefficients = extract_coefficients(lhs)
    return coefficients, operator, float(constant.strip())

def extract_coefficients(expr):
    """
    Extracts coefficients from a linear expression.
        expr (str): Linear expression, e.g., "2x + 3y".
    """
    coefficients = {}
    terms = re.findall(r"([+-]?\s*\d*\.?\d*)\s*([a-zA-Z_][a-zA-Z_0-9]*)", expr)
    for coef, var in terms:
        coef = coef.replace(" ", "")
        coefficients[var] = float(coef) if coef not in ("", "+", "-") else float(coef + "1")
    return coefficients

def parse_conditions(conditions):
    """
    Parses multiple constraints into matrix form.
        conditions (str): Multiline string of constraints.
    """
    inequalities = []
    variables = set()

    for line in conditions.strip().splitlines():
        coeffs, op, const = parse_constraint(line)
        inequalities.append((coeffs, op, const))
        variables.update(coeffs.keys())

    variables = sorted(variables)
    coeff_matrix, operators, constants = [], [], []

    for coeffs, op, const in inequalities:
        row = [coeffs.get(var, 0.0) for var in variables]
        coeff_matrix.append(row)
        operators.append(op)
        constants.append(const)

    return variables, coeff_matrix, operators, constants

def compare_values(lhs, rhs, operator):
    """
    Compares two values based on the specified operator.
        lhs (float): Left-hand side value.
        rhs (float): Right-hand side value.
        operator (str): Comparison operator.
    """
    operations = {
        "<": lambda x, y: x < y,
        ">": lambda x, y: x > y,
        "<=": lambda x, y: x <= y,
        ">=": lambda x, y: x >= y,
        "=": lambda x, y: x == y,
    }
    return operations[operator](lhs, rhs)

def find_finite_directions(opt_type, variables, coeffs, operators, bounds, target):
    """
    Determines finite directions for optimization feasibility.
        opt_type (str): Optimization type ('min' or 'max').
        variables (list): List of variables.
        coeffs (list): Coefficient matrix.
        operators (list): List of operators.
        bounds (list): List of constants.
        target (numpy.array): Target vector for the objective.
    """
    feasible_directions = []
    num_vars = len(variables)

    for rows in combinations(coeffs, num_vars - 1):
        lhs_matrix = list(rows) + [[1] * num_vars]
        rhs_vector = np.array([0] * (num_vars - 1) + [1]).reshape(-1, 1)
        try:
            solution = np.linalg.solve(lhs_matrix, rhs_vector).flatten()
        except np.linalg.LinAlgError:
            continue

        if all(compare_values(sum(row * solution), 0, op) for row, op in zip(coeffs, operators)):
            feasible_directions.append(solution)

    is_finite = all(
        (opt_type == "min" and np.dot(direction, target) >= 0) or
        (opt_type == "max" and np.dot(direction, target) <= 0)
        for direction in feasible_directions
    )
    return is_finite, feasible_directions

def calculate_vertices(variables, coeffs, operators, bounds):
    """
    Finds all feasible corner points.
        variables (list): List of variables.
        coeffs (list): Coefficient matrix.
        operators (list): List of operators.
        bounds (list): List of constants.
    """
    vertices = []
    num_vars = len(variables)

    for subset in combinations(zip(coeffs, bounds), num_vars):
        lhs, rhs = zip(*subset)
        try:
            solution = np.linalg.solve(lhs, rhs).flatten()
        except np.linalg.LinAlgError:
            continue

        if all(compare_values(sum(row * solution), bound, op) for row, op, bound in zip(coeffs, operators, bounds)):
            vertices.append(solution)
    return vertices

def optimize(objective, conditions):
    """
    Executes linear programming optimization.
        objective (str): Objective function.
        conditions (str): Constraints.
    """
    opt_type, obj_coeffs = parse_objective(objective)
    variables, coeff_matrix, ops, bounds = parse_conditions(conditions)

    if not set(obj_coeffs.keys()).issubset(set(variables)):
        raise ValueError("Objective function variables must appear in constraints.")

    target = np.array([obj_coeffs.get(var, 0.0) for var in variables]).reshape(-1, 1)

    is_finite, directions = find_finite_directions(opt_type, variables, coeff_matrix, ops, bounds, target)

    print("C:", target.flatten().tolist())
    print("Line directions:", [row for row in coeff_matrix])
    print("Has finite optimize value?", "Yes" if is_finite else "No")
    print("Apical recession directions:", [d.tolist() for d in directions])

    vertices = calculate_vertices(variables, coeff_matrix, ops, bounds)
    print("Corner points:", [v.tolist() for v in vertices])

    if not is_finite:
        return

    optimal_solutions, optimal_value = find_optimum(opt_type, target, vertices)
    print("optimize answers:", [sol.tolist() for sol in optimal_solutions])
    print("optimize value:", optimal_value.item() if isinstance(optimal_value, np.ndarray) else optimal_value)

def find_optimum(opt_type, target, vertices):
    """
    Finds the optimal solutions and value.
        opt_type (str): Optimization type ('min' or 'max').
        target (numpy.array): Target vector.
        vertices (list): Feasible corner points
    """
    opt_val = -np.inf if opt_type == "max" else np.inf
    optimal_solutions = []

    for vertex in vertices:
        value = np.dot(vertex, target).item()
        if (opt_type == "max" and value > opt_val) or (opt_type == "min" and value < opt_val):
            optimal_solutions = [vertex]
            opt_val = value
        elif value == opt_val:
            optimal_solutions.append(vertex)

    return optimal_solutions, opt_val

if __name__ == "__main__":
    
    obj1 = "max z = 2x + 3y"
    cond1 = """
    x + y <= 5
    x <= 3
    y <= 4
    x >= 0
    y >= 0
    """
    obj2 = "min z = 4x_1 - 2x_2 + x_3"
    cond2 = """
    2x_1 - x_2 + x_3 <= 10
    -x_1 + 3x_2 - 2x_3 <= 15
    x_1 >= 0
    x_2 >= 0
    x_3 >= 0
    """

    optimize(obj1, cond1)
    print("------------------------------")
    optimize(obj2, cond2)
    print("------------------------------")

