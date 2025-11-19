import csv
import re

INPUT_FILE = "FINAL_RESULTS_MMPBSA_PB.csv"

# Output filenames
OUTPUTS = {
    "Complex Energy Terms": "complex_energy_terms.csv",
    "Receptor Energy Terms": "receptor_energy_terms.csv",
    "Ligand Energy Terms": "ligand_energy_terms.csv",
    "Delta Energy Terms": "delta_energy_terms.csv",
}

def is_header_row(row):
    """Detects the numeric header line beginning with 'Frame #'."""
    return len(row) > 0 and row[0].strip().lower() == "frame #"

def is_data_row(row):
    """Check if a row starts with a number."""
    if len(row) == 0:
        return False
    try:
        int(float(row[0]))
        return True
    except:
        return False

def main():
    # Read entire CSV into memory
    with open(INPUT_FILE, "r", newline="") as f:
        csv_data = list(csv.reader(f))

    # Prepare storage for blocks
    blocks = {key: [] for key in OUTPUTS.keys()}
    current_block = None
    header_captured = False

    for row in csv_data:
        # Detect block title lines
        row_join = " ".join(cell.strip() for cell in row)

        for name in OUTPUTS.keys():
            if name in row_join:
                current_block = name
                header_captured = False
                break

        # Skip if no block selected yet
        if current_block is None:
            continue

        # Capture header
        if is_header_row(row) and not header_captured:
            blocks[current_block].append(row)
            header_captured = True
            continue

        # Capture data rows
        if header_captured and is_data_row(row):
            blocks[current_block].append(row)

    # Write each block to separate CSV
    for name, filename in OUTPUTS.items():
        data = blocks[name]
        if len(data) == 0:
            print(f"[WARNING] No data found for: {name}")
            continue

        with open(filename, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerows(data)

        print(f"[OK] Wrote {len(data)-1} rows → {filename}")

if __name__ == "__main__":
    main()