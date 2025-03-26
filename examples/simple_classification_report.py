#!/usr/bin/env python
"""
Simple script to demonstrate the separate-sentiment flag and display classifications
in a clean, readable format.
"""

import argparse
import json
import pandas as pd
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(
        description="Generate a clean classification report with separate sentiment results"
    )
    
    parser.add_argument("--urls", nargs="+", help="URLs to classify")
    parser.add_argument("--file", help="Input CSV file with a 'url' column")
    parser.add_argument("--output", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Construct the command to run the CLI with separate sentiment
    import subprocess
    import sys
    
    cmd = [sys.executable, "-m", "ad_intention.enhanced.cli", "--with-details", "--use-sentiment", "--separate-sentiment"]
    
    if args.urls:
        cmd.extend(["--urls"] + args.urls)
    elif args.file:
        cmd.extend(["--file", args.file])
    else:
        # Default example URLs
        cmd.extend(["--urls", 
                   "https://www.nike.com/", 
                   "https://www.microsoft.com/about",
                   "https://www.apple.com/shop/buy-mac", 
                   "https://www.airbnb.com/become-a-host",
                   "https://www.target.com/?utm_source=google"])
    
    # Add output if specified
    if args.output:
        cmd.extend(["--output", args.output])
    else:
        temp_output = "classification_temp.csv"
        cmd.extend(["--output", temp_output])
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Read the results
    output_file = args.output if args.output else "classification_temp.csv"
    df = pd.read_csv(output_file)
    
    # Process the results to create a clean table
    table_data = []
    
    for _, row in df.iterrows():
        url = row["url"]
        final_class = row["classification"]
        confidence = row["confidence"]
        
        # Parse classifications summary if available
        classifiers = {}
        if "classifications_summary" in row and row["classifications_summary"]:
            try:
                classifiers = json.loads(row["classifications_summary"])
            except:
                pass
        
        # Create a row for the table
        table_row = [
            url, 
            final_class, 
            f"{confidence:.2f}",
            classifiers.get("lexicon", "N/A"),
            classifiers.get("vader", "N/A"),
            classifiers.get("textblob", "N/A")
        ]
        
        table_data.append(table_row)
    
    # Display the table
    headers = ["URL", "Classification", "Confidence", "Lexicon", "VADER", "TextBlob"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Count agreement between different classifiers
    df_clean = pd.DataFrame(table_data, columns=headers)
    agreement_count = sum(
        (df_clean["Lexicon"] == df_clean["VADER"]) & 
        (df_clean["VADER"] == df_clean["TextBlob"]) &
        (df_clean["TextBlob"] != "N/A")
    )
    
    if len(df_clean) > 0 and "N/A" not in df_clean["TextBlob"].values:
        agreement_pct = agreement_count / len(df_clean) * 100
        print(f"\nAll classifiers agree: {agreement_count}/{len(df_clean)} ({agreement_pct:.1f}%)")
    
    # Cleanup temporary file
    if not args.output:
        import os
        try:
            os.remove(temp_output)
        except:
            pass

if __name__ == "__main__":
    main() 