import wandb
import pandas as pd
import os

# Your WandB entity (username/organization) and project name
ENTITY = "altonmaseko-wits-university"
PROJECT = "temporal-composition-gym"

def download_wandb_data():
    print(f"Connecting to WandB project: {ENTITY}/{PROJECT}...")
    api = wandb.Api()
    
    # Fetch all runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    print(f"Found {len(runs)} runs in the project.")

    # Create a directory to store the downloaded CSVs
    output_dir = "wandb_exports"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_list = []

    for run in runs:
        # Only process successfully finished runs
        if run.state != "finished":
            print(f"Skipping {run.name} (State: {run.state})")
            continue
            
        print(f"Downloading timeseries data for: {run.name}...")
        
        # 1. Save the run's final summary metrics and configuration
        summary_list.append({
            "run_id": run.id,
            "name": run.name,
            **run.summary._json_dict,  # Final metrics (e.g., final loss, final reward)
            **run.config               # Hyperparameters used
        })
        
        # 2. Download the full history (the timeseries data for the charts)
        # Using samples=1000 evenly samples the run to keep file sizes manageable.
        # If you want EVERY single step, change this to: history_df = run.history(pandas=(True))
        history_df = run.history(samples=1000) 
        
        # Save this run's timeseries data to its own CSV file
        safe_name = run.name.replace("/", "_").replace(":", "_")
        history_df.to_csv(os.path.join(output_dir, f"{safe_name}_history.csv"), index=False)

    # 3. Save the master summary file
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(output_dir, "all_runs_summary.csv"), index=False)
    
    print(f"\n✅ Success! All data has been saved to the '{output_dir}' folder.")
    print("-> 'all_runs_summary.csv' contains the final results and hyperparams for all 24 runs.")
    print("-> The other CSVs contain the step-by-step graph data (loss, rewards, etc.) for each run.")

if __name__ == "__main__":
    download_wandb_data()
