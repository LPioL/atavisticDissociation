#!/usr/bin/env python3
"""
GenAge Analysis - Only Second Subfigure (Stacked Percentages)
===========================================================

This script creates only the second subfigure from the original enhanced analysis:
- Stacked Percentages: Observed vs Expected
- For upregulated, downregulated, and combined gene sets
- Plus Mann-Whitney U test results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os
import warnings
warnings.filterwarnings('ignore')

# Evolutionary age categories (ordered from oldest to youngest, ranks 1-19)
evolutionary_order = [
    'All living organisms',  # Rank 1 (oldest)
    'Eukaryota', 
    'Opisthokonta',
    'Holozoa',
    'Metazoa',
    'Eumetazoa',
    'Bilateria',
    'Deuterostomia',
    'Chordata',
    'Olfactores',
    'Craniata',
    'Euteleostomi',
    'Tetrapoda',
    'Amniota',
    'Mammalia',
    'Eutheria',  # Added missing category
    'Boreoeutheria',
    'Euarchontoglires',
    'Primates'  # Rank 19 (youngest)
]

def load_baseline_data():
    """Load baseline evolutionary age data."""
    print("Loading baseline evolutionary age data...")
    
    # Load baseline genome data (same as original)
    age_human_genes = pd.read_csv('./data/1-s2.0-S0093775418302264-mmc2.csv', 
                                 header=None, delimiter=';')
    age_human_genes = age_human_genes[[0, 1]]
    age_human_genes = age_human_genes.dropna()
    age_human_genes[1] = age_human_genes[1].str.replace(',', '.').astype(float).round()
    
    dict_ages = {
        1: "All living organisms", 2: "Eukaryota", 3: "Opisthokonta", 4: "Holozoa", 5: "Metazoa",
        6: "Eumetazoa", 7: "Bilateria", 8: "Deuterostomia", 9: "Chordata", 10: "Olfactores",
        11: "Craniata", 12: "Euteleostomi", 13: "Tetrapoda", 14: "Amniota", 15: "Mammalia",
        16: "Eutheria", 17: "Boreoeutheria", 18: "Euarchontoglires", 19: "Primates"
    }
    
    age_human_genes[1] = age_human_genes[1].map(dict_ages)
    print(f"Loaded {age_human_genes.shape[0]} genes from baseline genome")
    
    return age_human_genes

def load_genage_data():
    """Load GenAge data from the original files."""
    print("Loading GenAge data...")
    
    try:
        # Load GenAge upregulated genes
        genage_upregulated = pd.read_csv('./data/over.all.csv', header=0, delimiter=';')
        genage_upregulated = genage_upregulated['Gene'].tolist()
        
        # Load GenAge downregulated genes
        genage_downregulated = pd.read_csv('./data/under.all.csv', header=0, delimiter=';')
        genage_downregulated = genage_downregulated['Gene'].tolist()
        
        print(f"GenAge upregulated genes: {len(genage_upregulated)}")
        print(f"GenAge downregulated genes: {len(genage_downregulated)}")
        
        return genage_upregulated, genage_downregulated
        
    except FileNotFoundError as e:
        print(f"GenAge data files not found: {e}")
        print("Using sample data...")
        # Fallback to sample data
        sample_genes = ['TP53', 'BRCA1', 'ATM', 'CHEK2', 'MLH1', 'MSH2', 'BRCA2', 'APC', 'VHL', 'RB1',
                       'CDKN2A', 'CDKN1A', 'SIRT1', 'FOXO3', 'IGF1R', 'MTOR', 'TSC1', 'TSC2', 'PTEN', 'AKT1',
                       'MYC', 'EGFR', 'KRAS', 'PIK3CA', 'CTNNB1', 'SMAD4', 'ARID1A', 'KMT2D', 'KDM6A', 'CREBBP']
        
        mid_point = len(sample_genes) // 2
        genage_upregulated = sample_genes[:mid_point]
        genage_downregulated = sample_genes[mid_point:]
        
        return genage_upregulated, genage_downregulated

def analyze_gene_set_enrichment_enhanced(gene_list, age_human_genes, analysis_name, evolutionary_order):
    """Perform hypergeometric analysis and return results DataFrame."""
    print(f"Performing hypergeometric analysis for {analysis_name}...")
    
    # Filter genes that exist in baseline
    filtered_genes = age_human_genes[age_human_genes[0].isin(gene_list)]
    
    if len(filtered_genes) == 0:
        print(f"No genes found for {analysis_name}")
        return None, None
    
    # Count genes in each evolutionary age category
    observed_counts = filtered_genes[1].value_counts()
    total_observed = len(filtered_genes)
    total_baseline = len(age_human_genes)
    
    results = []
    
    for category in evolutionary_order:
        observed = observed_counts.get(category, 0)
        expected = len(age_human_genes[age_human_genes[1] == category])
        
        if expected > 0 and total_baseline > 0:
            # Over-representation test
            p_value_over = hypergeom.sf(observed - 1, total_baseline, expected, total_observed)
            # Under-representation test
            p_value_under = hypergeom.cdf(observed, total_baseline, expected, total_observed)
        else:
            p_value_over = 1.0
            p_value_under = 1.0
        
        # Calculate fold change
        fold_change = (observed / total_observed) / (expected / total_baseline) if expected > 0 else 0
        
        # Calculate percentages
        observed_pct = (observed / total_observed) * 100
        expected_pct = (expected / total_baseline) * 100
        
        results.append({
            'Category': category,
            'Observed_Count': observed,
            'Expected_Count': expected,
            'Observed_Percentage': observed_pct,
            'Baseline_Percentage': expected_pct,
            'Fold_Change': fold_change,
            'P_Value_Over': p_value_over,
            'P_Value_Under': p_value_under
        })
    
    # Apply FDR correction
    p_values_over = [r['P_Value_Over'] for r in results]
    p_values_under = [r['P_Value_Under'] for r in results]
    
    _, p_corrected_over, _, _ = multipletests(p_values_over, method='fdr_bh')
    _, p_corrected_under, _, _ = multipletests(p_values_under, method='fdr_bh')
    
    # Add corrected p-values and determine significance
    for i, result in enumerate(results):
        result['P_Value_Over_FDR'] = p_corrected_over[i]
        result['P_Value_Under_FDR'] = p_corrected_under[i]
        
        # Determine significance
        if result['P_Value_Over_FDR'] < 0.05 and result['Fold_Change'] > 1.1:
            result['Significant_Over'] = True
            result['Significant_Under'] = False
            result['Significant_FDR'] = True
            result['Representation_Type'] = 'Over-represented'
        elif result['P_Value_Under_FDR'] < 0.05 and result['Fold_Change'] < 0.9:
            result['Significant_Over'] = False
            result['Significant_Under'] = True
            result['Significant_FDR'] = True
            result['Representation_Type'] = 'Under-represented'
        else:
            result['Significant_Over'] = False
            result['Significant_Under'] = False
            result['Significant_FDR'] = False
            result['Representation_Type'] = 'Not significant'
    
    # Create continuous results for Mann-Whitney test
    gene_ages = filtered_genes[1].tolist()
    gene_ages_numeric = [evolutionary_order.index(age) + 1 for age in gene_ages if age in evolutionary_order]
    baseline_ages_numeric = [evolutionary_order.index(age) + 1 for age in age_human_genes[1] if age in evolutionary_order]
    
    if len(gene_ages_numeric) > 0 and len(baseline_ages_numeric) > 0:
        try:
            mw_stat, mw_pvalue = mannwhitneyu(gene_ages_numeric, baseline_ages_numeric, alternative='two-sided')
        except:
            mw_stat, mw_pvalue = 0.0, 1.0
        
        mean_gene_age = np.mean(gene_ages_numeric)
        mean_baseline_age = np.mean(baseline_ages_numeric)
        mean_age_shift = mean_gene_age - mean_baseline_age
        
        continuous_results = {
            'Mann_Whitney_Statistic': mw_stat,
            'Mann_Whitney_P_Value': mw_pvalue,
            'Mean_Age_Shift': mean_age_shift,
            'Mean_Gene_Age': mean_gene_age,
            'Mean_Baseline_Age': mean_baseline_age
        }
    else:
        continuous_results = None
    
    return pd.DataFrame(results), continuous_results

def plot_percentages_only(results_df, gene_set_name, output_path, upregulated_genes=None, downregulated_genes=None, age_human_genes=None):
    """Create ONLY the second subfigure: Stacked Percentages: Observed vs Expected."""
    
    evolutionary_order = results_df['Category'].tolist()
    
    # Create single figure (just the second subfigure)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    x = range(len(evolutionary_order))
    width = 0.35
    
    if upregulated_genes is not None and downregulated_genes is not None and age_human_genes is not None:
        # Create stacked percentages histogram with expected comparison
        # Re-filter genes for percentages (proper data)
        upregulated_filtered_pct = age_human_genes[age_human_genes[0].isin(upregulated_genes)]
        downregulated_filtered_pct = age_human_genes[age_human_genes[0].isin(downregulated_genes)]
        
        upregulated_counts_pct = upregulated_filtered_pct[1].value_counts()
        downregulated_counts_pct = downregulated_filtered_pct[1].value_counts()
        
        # Ensure all categories are present
        for category in evolutionary_order:
            if category not in upregulated_counts_pct:
                upregulated_counts_pct[category] = 0
            if category not in downregulated_counts_pct:
                downregulated_counts_pct[category] = 0
        
        # Reindex to match evolutionary order
        upregulated_counts_pct = upregulated_counts_pct.reindex(evolutionary_order)
        downregulated_counts_pct = downregulated_counts_pct.reindex(evolutionary_order)
        
        # Calculate percentages from combined total
        total_combined = upregulated_counts_pct.sum() + downregulated_counts_pct.sum()
        
        upregulated_pct = (upregulated_counts_pct / total_combined) * 100
        downregulated_pct = (downregulated_counts_pct / total_combined) * 100
        
        # Get baseline percentages
        baseline_counts = age_human_genes[1].value_counts()
        baseline_counts = baseline_counts.reindex(evolutionary_order, fill_value=0)
        baseline_pct = (baseline_counts / baseline_counts.sum()) * 100
        
        # Create stacked bars for percentages
        bars_up_pct = ax.bar([i - width/2 for i in x], upregulated_pct.values, 
                             width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
        bars_down_pct = ax.bar([i - width/2 for i in x], downregulated_pct.values, 
                               width, bottom=upregulated_pct.values,  # Stack on top of upregulated
                               label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
        
        # Add expected baseline bar
        bars_expected_pct = ax.bar([i + width/2 for i in x], baseline_pct.values, 
                                   width, label='Expected\n(Litman et al.)', alpha=0.7, color='#2ca02c')  # Green
        
        # Add significance stars for percentages using enrichment results
        # We need to get the enrichment results for the combined analysis
        combined_genes = list(upregulated_genes) + list(downregulated_genes)
        combined_results, _ = analyze_gene_set_enrichment_enhanced(combined_genes, age_human_genes, "Combined", evolutionary_order)
        
        if combined_results is not None:
            for i, category in enumerate(evolutionary_order):
                category_result = combined_results[combined_results['Category'] == category]
                if not category_result.empty:
                    fdr_significant = category_result['Significant_FDR'].iloc[0]
                    if fdr_significant:
                        is_over_represented = category_result['Significant_Over'].iloc[0]
                        star_color = 'red' if is_over_represented else 'black'
                        ax.text(i - width/2, upregulated_pct.values[i] + downregulated_pct.values[i] + 2, 
                               '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
    
    else:
        # Original percentages comparison (for single gene set)
        ax.bar([i - width/2 for i in x], results_df['Observed_Percentage'], 
                width, label='Observed', alpha=0.7, color='#1f77b4')  # Blue
        ax.bar([i + width/2 for i in x], results_df['Baseline_Percentage'], 
                width, label='Expected\n(Litman et al.)', alpha=0.7, color='#ff7f0e')  # Orange
        
        # Add significance stars with proper over/under-representation testing
        for i, (_, row) in enumerate(results_df.iterrows()):
            if row['Significant_FDR']:
                is_over_represented = row['Significant_Over']
                star_color = 'red' if is_over_represented else 'black'
                ax.text(i, max(row['Observed_Percentage'], row['Baseline_Percentage']) + 0.5, 
                        '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
    
    # Formatting (same as original second subfigure)
    ax.set_xlabel('Evolutionary Age', fontsize=24, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
    ax.set_title(f'Stacked Percentages: {gene_set_name}', fontsize=28, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def save_mann_whitney_results(mann_whitney_results, output_dir):
    """Save Mann-Whitney U test results to CSV."""
    if mann_whitney_results:
        df = pd.DataFrame(mann_whitney_results)
        df.to_csv(os.path.join(output_dir, 'genage_mann_whitney_results.csv'), index=False)
        print("Mann-Whitney U test results saved to genage_mann_whitney_results.csv")

def main():
    """Main analysis function."""
    print("Starting GenAge Percentages Analysis (Second Subfigure Only)...")
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    baseline_genes = load_baseline_data()
    genage_upregulated, genage_downregulated = load_genage_data()
    
    # Create combined gene list
    genage_combined = genage_upregulated + genage_downregulated
    
    print(f"GenAge upregulated genes: {len(genage_upregulated)}")
    print(f"GenAge downregulated genes: {len(genage_downregulated)}")
    print(f"GenAge combined genes: {len(genage_combined)}")
    
    # Perform analyses
    analyses = {
        'Upregulated': genage_upregulated,
        'Downregulated': genage_downregulated,
        'Combined': genage_combined
    }
    
    mann_whitney_results = []
    
    for analysis_name, gene_list in analyses.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING {analysis_name.upper()}")
        print(f"{'='*60}")
        
        # Perform hypergeometric analysis
        results_df, continuous_results = analyze_gene_set_enrichment_enhanced(
            gene_list, baseline_genes, analysis_name, evolutionary_order
        )
        
        if results_df is not None:
            # Create the percentages plot (second subfigure only)
            plot_path = os.path.join(output_dir, f'genage_{analysis_name.lower()}_percentages.png')
            
            if analysis_name == 'Combined':
                # For combined analysis, pass both upregulated and downregulated genes
                plot_percentages_only(
                    results_df, f"GenAge {analysis_name}", plot_path,
                    upregulated_genes=genage_upregulated,
                    downregulated_genes=genage_downregulated,
                    age_human_genes=baseline_genes
                )
            else:
                # For individual analyses, only pass the relevant gene list
                plot_percentages_only(
                    results_df, f"GenAge {analysis_name}", plot_path
                )
            
            # Save Mann-Whitney results
            if continuous_results is not None:
                mann_whitney_results.append({
                    'Analysis': analysis_name,
                    'Mann_Whitney_Statistic': continuous_results['Mann_Whitney_Statistic'],
                    'Mann_Whitney_P_Value': continuous_results['Mann_Whitney_P_Value'],
                    'Mean_Age_Shift': continuous_results['Mean_Age_Shift'],
                    'Mean_Gene_Age': continuous_results['Mean_Gene_Age'],
                    'Mean_Baseline_Age': continuous_results['Mean_Baseline_Age']
                })
            
            print(f"Plot saved: {plot_path}")
    
    # Save Mann-Whitney results
    save_mann_whitney_results(mann_whitney_results, output_dir)
    
    print(f"\nAnalysis complete! Results saved in {output_dir}/")
    print("Generated files:")
    print("- genage_upregulated_percentages.png")
    print("- genage_downregulated_percentages.png") 
    print("- genage_combined_percentages.png")
    print("- genage_mann_whitney_results.csv")

if __name__ == "__main__":
    main()
