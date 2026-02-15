#!/usr/bin/env python3
"""
Interactive Optimal Superplasticizer Dosage Predictor
Reads input from the user via prompts.

Usage:
    python run_predictor.py
"""

from predict_dosage import load_model, predict_optimal_dosage


def get_wc_ratio():
    """Prompt the user for a valid W/C ratio."""
    while True:
        try:
            wc = float(input("Enter Water/Cement ratio (e.g., 0.35, 0.40, 0.45): "))
            if wc <= 0:
                print("❌ W/C ratio must be a positive number. Try again.")
                continue
            return wc
        except ValueError:
            print("❌ Invalid input. Please enter a numeric value.")


def get_sp_type():
    """Prompt the user for a valid superplasticizer type."""
    while True:
        sp = input("Enter Superplasticizer type (PCE or SNF): ").strip().upper()
        if sp in ("PCE", "SNF"):
            return sp
        print("❌ Invalid type. Please enter 'PCE' or 'SNF'.")


def main():
    print("\n" + "=" * 60)
    print("  OPTIMAL SUPERPLASTICIZER DOSAGE PREDICTOR")
    print("=" * 60)

    wc_ratio = get_wc_ratio()
    sp_type = get_sp_type()
    verbose_input = input("Show detailed information? (y/n): ").strip().lower()
    verbose = verbose_input in ("y", "yes")

    # Load model and predict
    model, label_encoder = load_model()
    optimal_dosage = predict_optimal_dosage(wc_ratio, sp_type, model, label_encoder)

    # Display results
    print("\n" + "=" * 60)
    print("  PREDICTION RESULTS")
    print("=" * 60)
    print(f"  Water/Cement Ratio:       {wc_ratio}")
    print(f"  Superplasticizer Type:    {sp_type}")
    print(f"  Predicted Optimal Dosage: {optimal_dosage:.3f}%")
    print("=" * 60)

    if verbose:
        print("\nInterpretation:")
        print(f"  • Use {optimal_dosage:.3f}% {sp_type} by weight of cement")
        print(f"  • This is the saturation point for W/C ratio {wc_ratio}")
        print(f"  • Adding more than this amount will not improve fluidity")

        if sp_type == "PCE":
            print(f"\n  Note: PCE typically requires 0.35-0.45% dosage")
        else:
            print(f"\n  Note: SNF typically requires 0.40-0.50% dosage")

    print()


if __name__ == "__main__":
    main()
