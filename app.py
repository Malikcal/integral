from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sympy as sp
from sympy import (
    symbols, integrate, diff, simplify, latex, sympify,
    sin, cos, tan, exp, log, sqrt, pi, oo, E,
    sec, csc, cot, asin, acos, atan, sinh, cosh, tanh,
    Rational, Symbol, Function
)
import os, re

app = Flask(__name__, static_folder=".")
CORS(app)

SAFE_LOCALS = {
    "x": symbols("x"), "y": symbols("y"), "z": symbols("z"),
    "t": symbols("t"), "n": symbols("n"), "a": symbols("a"),
    "b": symbols("b"), "c": symbols("c"),
    "sin": sin, "cos": cos, "tan": tan, "cot": cot,
    "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan,
    "arcsin": asin, "arccos": acos, "arctan": atan,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "exp": exp, "log": log, "ln": log,
    "sqrt": sqrt, "pi": pi, "e": E, "E": E,
    "oo": oo, "inf": oo,
    "Abs": sp.Abs, "abs": sp.Abs,
}


def preprocess(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)
    expr = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", expr)
    expr = expr.replace("^", "**")
    expr = expr.replace("ln(", "log(")
    return expr


def parse_expr(expr_str: str, var_sym):
    processed = preprocess(expr_str)
    local = {**SAFE_LOCALS, str(var_sym): var_sym}
    return sympify(processed, locals=local)


def build_steps_indefinite(expr, integral_result, var_sym, original_str):
    var_latex = latex(var_sym)
    steps = []

    steps.append({
        "title": "Write the Integral",
        "description": f"We need to evaluate the following indefinite integral:",
        "latex": f"\\int {latex(expr)} \\, d{var_latex}"
    })

    integral_type, rule_name, rule_desc = identify_rule(expr, var_sym)

    steps.append({
        "title": f"Identify the Rule: {rule_name}",
        "description": rule_desc,
        "latex": None
    })

    intermediate_steps = get_intermediate_steps(expr, var_sym, integral_type)
    steps.extend(intermediate_steps)

    simplified = simplify(integral_result)

    steps.append({
        "title": "Apply the Integration Rule",
        "description": "Applying the identified rule and integrating term by term:",
        "latex": f"\\int {latex(expr)} \\, d{var_latex} = {latex(simplified)} + C"
    })

    steps.append({
        "title": "Simplify the Result",
        "description": "Simplify and write the final answer with the constant of integration C:",
        "latex": f"= {latex(simplified)} + C"
    })

    return steps, simplified


def build_steps_definite(expr, result, var_sym, lower, upper):
    var_latex = latex(var_sym)
    antideriv = integrate(expr, var_sym)
    antideriv_simplified = simplify(antideriv)
    steps = []

    steps.append({
        "title": "Write the Definite Integral",
        "description": "We need to evaluate the following definite integral:",
        "latex": f"\\int_{{{latex(lower)}}}^{{{latex(upper)}}} {latex(expr)} \\, d{var_latex}"
    })

    steps.append({
        "title": "Find the Antiderivative",
        "description": "First, find the indefinite integral (antiderivative):",
        "latex": f"F({var_latex}) = \\int {latex(expr)} \\, d{var_latex} = {latex(antideriv_simplified)} + C"
    })

    steps.append({
        "title": "Apply the Fundamental Theorem of Calculus",
        "description": "Evaluate F at the upper and lower bounds and subtract:",
        "latex": f"\\int_{{{latex(lower)}}}^{{{latex(upper)}}} {latex(expr)} \\, d{var_latex} = F({latex(upper)}) - F({latex(lower)})"
    })

    try:
        upper_val = antideriv_simplified.subs(var_sym, upper)
        lower_val = antideriv_simplified.subs(var_sym, lower)
        steps.append({
            "title": "Substitute the Bounds",
            "description": f"Substitute x = {latex(upper)} and x = {latex(lower)}:",
            "latex": (
                f"= \\left[{latex(antideriv_simplified)}\\right]_{{{latex(lower)}}}^{{{latex(upper)}}} "
                f"= \\left({latex(simplify(upper_val))}\\right) - \\left({latex(simplify(lower_val))}\\right)"
            )
        })
    except Exception:
        pass

    simplified_result = simplify(result)
    steps.append({
        "title": "Compute the Final Value",
        "description": "Subtract to get the definite integral value:",
        "latex": f"= {latex(simplified_result)}"
    })

    return steps, simplified_result


def identify_rule(expr, var_sym):
    expr_str = str(expr)
    v = str(var_sym)

    if expr == var_sym:
        return "power", "Power Rule", f"For ∫x dx, use the Power Rule: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C"
    if expr.is_number:
        return "constant", "Constant Rule", "For a constant k, ∫k dx = kx + C"
    if expr.is_Add:
        return "sum", "Sum/Difference Rule", "Break the integral into individual terms: ∫(f ± g) dx = ∫f dx ± ∫g dx"
    if expr.is_Mul:
        args = expr.args
        if any(a.is_number for a in args):
            return "const_mul", "Constant Multiple Rule", "Factor out the constant: ∫k·f(x) dx = k·∫f(x) dx"
        return "product", "Integration by Parts / Substitution", "For products, use integration by parts: ∫u dv = uv − ∫v du"
    if expr.is_Pow:
        base, exp_val = expr.args
        if base == var_sym:
            if exp_val == -1:
                return "log", "Logarithm Rule", "∫(1/x) dx = ln|x| + C"
            return "power", "Power Rule", f"For ∫x^n dx, use the Power Rule: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C, where n ≠ −1"
        if base == sp.E or str(base) == "E" or str(base) == "e":
            return "exp", "Exponential Rule", f"∫eˣ dx = eˣ + C; for ∫e^(ax) dx = e^(ax)/a + C"
    if expr.func == sp.exp or (expr.is_Pow and expr.args[0] == sp.E):
        return "exp", "Exponential Rule", "∫e^(ax) dx = e^(ax)/a + C"
    if expr.func in (sin, cos, tan, sec, csc, cot):
        trig_rules = {
            sin: "∫sin(x) dx = −cos(x) + C",
            cos: "∫cos(x) dx = sin(x) + C",
            tan: "∫tan(x) dx = −ln|cos(x)| + C",
            sec: "∫sec(x) dx = ln|sec(x) + tan(x)| + C",
            csc: "∫csc(x) dx = −ln|csc(x) + cot(x)| + C",
            cot: "∫cot(x) dx = ln|sin(x)| + C",
        }
        rule = trig_rules.get(expr.func, "Standard trigonometric integral rule")
        return "trig", "Trigonometric Rule", rule
    if expr.func in (sinh, cosh, tanh):
        return "hyp", "Hyperbolic Rule", "Use standard hyperbolic integral formulas."
    if expr.func == log:
        return "log_func", "Integration by Parts (ln x)", "∫ln(x) dx = x·ln(x) − x + C, derived via integration by parts."

    return "general", "Standard Integration Techniques", "Apply substitution or integration by parts as needed."


def get_intermediate_steps(expr, var_sym, integral_type):
    steps = []
    v = latex(var_sym)

    if integral_type == "sum":
        terms = sp.Add.make_args(expr)
        term_integrals = []
        for t in terms:
            ti = integrate(t, var_sym)
            term_integrals.append(f"\\int {latex(t)} \\, d{v} = {latex(simplify(ti))}")
        steps.append({
            "title": "Split into Individual Terms",
            "description": "Integrate each term separately using the Sum Rule:",
            "latex": " \\\\ ".join(term_integrals)
        })

    elif integral_type == "const_mul":
        args = expr.args
        const = [a for a in args if a.is_number]
        func  = [a for a in args if not a.is_number]
        if const and func:
            c_val = const[0]
            f_val = sp.Mul(*func)
            steps.append({
                "title": "Factor Out the Constant",
                "description": "Pull the constant coefficient outside the integral:",
                "latex": f"= {latex(c_val)} \\int {latex(f_val)} \\, d{v}"
            })

    elif integral_type == "power":
        if expr.is_Pow:
            base, exp_val = expr.args
            steps.append({
                "title": "Apply the Power Rule",
                "description": f"For ∫xⁿ dx = xⁿ⁺¹/(n+1) + C, here n = {latex(exp_val)}:",
                "latex": f"\\int {latex(base)}^{{{latex(exp_val)}}} \\, d{v} = \\frac{{{latex(base)}^{{{latex(exp_val + 1)}}}}}{{{latex(exp_val + 1)}}} + C"
            })

    elif integral_type == "exp":
        steps.append({
            "title": "Apply the Exponential Rule",
            "description": "For exponential functions, the integral of eˣ is eˣ itself. For e^(ax), divide by a:",
            "latex": f"\\int e^{{ax}} \\, d{v} = \\frac{{e^{{ax}}}}{{a}} + C"
        })

    elif integral_type == "trig":
        trig_table = {
            "sin": f"\\int \\sin({v}) \\, d{v} = -\\cos({v}) + C",
            "cos": f"\\int \\cos({v}) \\, d{v} = \\sin({v}) + C",
            "tan": f"\\int \\tan({v}) \\, d{v} = -\\ln|\\cos({v})| + C",
        }
        name = expr.func.__name__
        if name in trig_table:
            steps.append({
                "title": "Recall the Trigonometric Integral Formula",
                "description": "Using the standard trigonometric integration formula:",
                "latex": trig_table[name]
            })

    return steps


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/integrate", methods=["POST"])
def do_integrate():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided."}), 400

    func_str    = data.get("function", "").strip()
    var_str     = data.get("variable", "x").strip() or "x"
    lower_str   = data.get("lower", "").strip()
    upper_str   = data.get("upper", "").strip()
    is_definite = lower_str != "" and upper_str != ""

    if not func_str:
        return jsonify({"success": False, "error": "Please enter a function to integrate."}), 200

    try:
        var_sym = symbols(var_str)
        expr = parse_expr(func_str, var_sym)
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not parse expression: {e}"}), 200

    try:
        if is_definite:
            lower_sym = parse_expr(lower_str, var_sym)
            upper_sym = parse_expr(upper_str, var_sym)
            result = integrate(expr, (var_sym, lower_sym, upper_sym))
            steps, simplified = build_steps_definite(expr, result, var_sym, lower_sym, upper_sym)
            final_latex = latex(simplified)
            is_indef = False
        else:
            result = integrate(expr, var_sym)
            steps, simplified = build_steps_indefinite(expr, result, var_sym, func_str)
            final_latex = f"{latex(simplified)} + C"
            is_indef = True

        return jsonify({
            "success": True,
            "inputLatex": f"\\int {latex(expr)} \\, d{latex(var_sym)}" if not is_definite else f"\\int_{{{latex(parse_expr(lower_str, var_sym))}}}^{{{latex(parse_expr(upper_str, var_sym))}}} {latex(expr)} \\, d{latex(var_sym)}",
            "interpretedLatex": latex(expr),
            "finalLatex": final_latex,
            "isIndefinite": is_indef,
            "steps": steps,
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Integration failed: {e}"}), 200


@app.route("/api/preview", methods=["POST"])
def preview():
    data = request.get_json() or {}
    func_str = data.get("function", "").strip()
    var_str  = data.get("variable", "x").strip() or "x"
    if not func_str:
        return jsonify({"latex": ""})
    try:
        var_sym = symbols(var_str)
        expr = parse_expr(func_str, var_sym)
        return jsonify({"latex": latex(expr), "success": True})
    except Exception:
        return jsonify({"latex": func_str, "success": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Integral Calculator running at http://localhost:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
