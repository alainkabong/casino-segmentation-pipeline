import os
import sys
import glob
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================
OUTPUT_DIR = r"C:\Users\MwambKabong\Downloads\Casino_Segementation_Pipeline\artifacts\batch_predictions"


CUSTOMER_IDS = []


def find_latest_segment_assignments(output_dir):
    
    pattern = os.path.join(output_dir, "segment_assignments_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No segment_assignments_*.csv found in {output_dir}. "
            f"Run Python_Batch_Prediction_Test_Code.py at least once first -- "
            f"this script only reads an already-saved output, it doesn't compute anything itself.")
    
    latest = sorted(matches)[-1]
    return latest


def lookup(customer_ids, output_dir=OUTPUT_DIR):
    csv_path = find_latest_segment_assignments(output_dir)
    mtime = pd.Timestamp.fromtimestamp(os.path.getmtime(csv_path))
    print(f"Reading: {os.path.basename(csv_path)}")
    print(f"(saved {mtime.strftime('%Y-%m-%d %H:%M:%S')} -- this lookup is only as fresh as that run)")
    print()

    df = pd.read_csv(csv_path, dtype={"segment_id": "Int64"})   # whole numbers even though excluded customers leave it blank
    found_ids = set(df["Customer_ID"])

    results = []
    for cid in customer_ids:
        cid = int(cid)
        if cid not in found_ids:
            print(f"Customer_ID {cid}: NOT FOUND in this extract.")
            print(f"Either they weren't in the population this file was built from, or they're")
            print(f"new since this run -- re-run Python_Batch_Prediction_Test_Code.py on current")
            print(f"data first if you need them, this script can't create data that isn't here.")
            print()
            continue

        row = df[df["Customer_ID"] == cid].iloc[0]
        print(f"{'='*70}")
        print(f"Customer_ID: {cid}")
        print(f"{'='*70}")
        # Customers excluded from segmentation (e.g. the zero-cash-funded cohort) have no segment, scores or
        # bonus fields at all -- show why, instead of a page of "nan".
        exclusion_reason = row.get("segment_exclusion_reason")
        if row.get("is_excluded") in (True, "True") or pd.notna(exclusion_reason):
           
            exclusion_code = row.get("exclusion_reason_code")
            print(f"is_excluded: True")
            print(f"exclusion_reason_code: "
                  f"{exclusion_code if pd.notna(exclusion_code) else 'not in this file (produced before this field existed)'}")
            print(f"Segment: none -- not part of any segment or sub-segment")
            print(f"Reason: {exclusion_reason}")
            print()
            print(f"Deserves bonus: NO -- {row.get('amount_basis')}")
            print()
            results.append(row)
            continue
        print(f"Segment: {row.get('segment_id')} -- {row.get('persona')}")
        print(f"Sub-segment: {row.get('sub_segment_id')} -- {row.get('sub_persona')}")
        print(f"Confidence: {row.get('assignment_confidence')}")
        print()
        print(f"Bonus worthiness score: {row.get('bonus_worthiness_score')}  "
              f"(value={row.get('value_score')}, recency={row.get('recency_score')}, "
              f"responsiveness={row.get('responsiveness_score')}, engagement={row.get('engagement_score')})")
        # PCA-derived weights behind the four scores above -- one value per batch RUN, not per customer,
        # so every segmented customer in this file shows the same weights here.
        weights = row.get("worthiness_weights")
        if pd.notna(weights):
            print(f"Component weights (PCA-derived for this run): {weights}")
        print()
        if row.get("deserves_bonus") in (True, "True"):
            print(f"Deserves bonus:   YES")
            print(f"Worthiness tier: {row.get('worthiness_tier')}")
            print(f"Bonus amount: ${row.get('bonus_amount_policy')}")
            print(f"Business rule component: ${row.get('bonus_amount_business_rule')}")
            print(f"Profitability rule component: ${row.get('bonus_amount_profitability_rule')}")
            print(f"Segment-aware coefficient: {row.get('segment_multiplier')}x  "
                  f"(-> ${row.get('bonus_amount_segment_rule')})")
            print()
            
            print(f"Customer NGR (last window): ${row.get('customer_ngr')}   "
                  f"Theoretical GGR (turnover x 3%): ${row.get('theoretical_ggr')}   "
                  f"Average stake: ${row.get('customer_avg_stake')}")
            ngr_ratio, stake_ratio = row.get("bonus_to_ngr_ratio"), row.get("bonus_to_stake_ratio")
            print(f"Bonus / NGR: {ngr_ratio if pd.notna(ngr_ratio) else 'n/a (NGR not positive)'}   "
                  f"Bonus / avg stake: {stake_ratio if pd.notna(stake_ratio) else 'n/a (no stake on record)'}")
            
            product_type = row.get("bonus_product_type")
            if pd.notna(product_type):
               
                print(f"  (approved amount in MT: {row.get('bonus_amount_policy_mt')})")
                print(f"Structured as: {row.get('bonus_flight_count')} x {row.get('bonus_product_type')} "
                      f"@ {row.get('bonus_flight_size')} each (delivers {row.get('bonus_structured_total')})")
            elif pd.notna(row.get("bonus_structure_note")):
                print(f"Not structured into flights: {row.get('bonus_structure_note')}")
        else:
            print(f"Deserves bonus: NO -- {row.get('amount_basis')}")
        print()
        results.append(row)

    return results


if __name__ == "__main__":
    ids = sys.argv[1:] if len(sys.argv) > 1 else CUSTOMER_IDS
    if not ids:
        print("No customer IDs given -- edit CUSTOMER_IDS in this script, or pass them as arguments:")
        print("python bonus_lookup.py 12623134 87654321")
    else:
        lookup(ids)
