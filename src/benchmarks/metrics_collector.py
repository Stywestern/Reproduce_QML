############################################################################################
#                                       Imports
############################################################################################

# Native Libraries
import re

############################################################################################
#                                       Main Block
############################################################################################

class MetricsCollector:
    """Centralized formatting and payload builder for algorithm benchmarking."""

    @staticmethod
    def _natural_key(key_name):
        """
        Splits variable names into string and integer chunks for natural sorting.
        e.g., 'x10' -> ('x', 10), 'x2' -> ('x', 2), 'alpha' -> ('alpha', 0)
        Guarantees no comparison between str and int.
        """
        s_key = str(key_name)
        match = re.match(r"^([a-zA-Z_]+)(\d+)$", s_key)
        if match:
            prefix, number = match.groups()
            return (prefix, int(number))
        return (s_key, 0)

    @classmethod
    def _extract_bitstring(cls, best_sols: list) -> str:
        """Helper to extract a clean binary bitstring (e.g., '110101') from solutions."""
        if not best_sols:
            return "N/A"
        
        formatted_bitstrings = []
        for sol in best_sols:
            # Bulletproof natural sorting using _natural_key
            sorted_keys = sorted(sol.keys(), key=cls._natural_key)
            bitstring = "".join(str(sol[k]) for k in sorted_keys)
            formatted_bitstrings.append(bitstring)
            
        return ", ".join(formatted_bitstrings)

    @classmethod
    def format_results(cls, paradigm_name: str, best_cost: float, best_sols: list, execution_time: float, raw_details: dict):
        """Constructs standardized metrics dictionaries and handles logging in priority order."""
        
        is_feasible = best_sols is not None and len(best_sols) > 0
        validity = f"Global Optimum Found (Cost: {best_cost})" if is_feasible else "Infeasible State Hit"
        solution_bitstring = cls._extract_bitstring(best_sols)

        # Standardized Payload
        metrics = {
            "paradigm": paradigm_name,
            "solution_bitstring": solution_bitstring,
            "best_cost": best_cost,
            "execution_time_seconds": execution_time,
            "validity": validity,
            "best_solutions": best_sols,
            "paradigm_details": raw_details
        }

        # Centralized Print Formatting (Strict Priority Order: 1. Bitstring | 2. Cost | 3. Time | 4. Details)
        print(f"==================================================")
        print(f"  RUN SUMMARY: {paradigm_name.upper()}")
        print(f"==================================================")
        print(f"  * [1] Solution Bitstring   : {solution_bitstring}")
        print(f"  * [2] Final Objective Cost : {best_cost}")
        print(f"  * [3] Execution Time       : {execution_time:.6f} s")
        print(f"  * [4] Status / Validity    : {validity}")
        print(f"  --------------------------------------------------")
        print(f"  * PARADIGM DETAILS:")
        for key, val in raw_details.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"    - {formatted_key:<22} : {val}")
        print(f"==================================================\n")

        return metrics

    @staticmethod
    def print_benchmark_table(all_metrics: list[dict]):
        """
        Prints a unified, formatted ASCII benchmark comparison table across all evaluated paradigms.
        """
        print("\n" + "=" * 110)
        print(" " * 38 + "BENCHMARK SHOWDOWN COMPARISON TABLE")
        print("=" * 110)
        
        headers = [
            "Paradigm / Engine", 
            "Bitstring", 
            "Cost", 
            "Time (s)", 
            "Qubits / Vars", 
            "Slack Qubits",
            "Validity / Status"
        ]
        
        header_row = (
            f"| {headers[0]:<35} | {headers[1]:<10} | {headers[2]:<8} | "
            f"{headers[3]:<10} | {headers[4]:<13} | {headers[6]:<15} |"
        )
        print(header_row)
        print("-" * 110)

        for m in all_metrics:
            paradigm = m.get("paradigm", "Unknown")[:35]
            bitstring = m.get("solution_bitstring", "N/A")[:10]
            cost = f"{m.get('best_cost', 'N/A'):.2f}" if isinstance(m.get('best_cost'), (int, float)) else "N/A"
            time_s = f"{m.get('execution_time_seconds', 0):.6f}"
            
            # Extract Qubit / Slack data dynamically
            details = m.get("paradigm_details", {})
            qubits = (
                details.get("qubits_required") 
                or details.get("logical_qubits") 
                or details.get("variables_evaluated") 
                or details.get("num_variables")
                or "N/A"
            )
            validity = "VALID" if "Global Optimum" in m.get("validity", "") or "Feasible" in m.get("validity", "") else "INVALID"

            row = (
                f"| {paradigm:<35} | {bitstring:<10} | {cost:<8} | "
                f"{time_s:<10} | {str(qubits):<13} | {validity:<15} |"
            )
            print(row)

        print("=" * 110 + "\n")