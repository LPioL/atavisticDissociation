

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellAge Signature Analysis v2 - Enhanced with Raw Counts and Under-representation
Following the exact code structure from analysis_atavism_combined.py

This script analyzes cellAge signature data with:
1. Original analysis (raw counts) - as in the original code
2. Improved analysis (percentages, cumulative hypergeometric tests) - addressing reviewer concerns
3. Enhanced features: raw counts, stacked histograms, under-representation tests

Author:  llopez06)
"""

import numpy as np
import pandas as pd
import math
from collections import Counter
from pyarrow import table
from scipy.spatial import distance
import networkx as nx
import requests
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from datetime import datetime 
import h5py
import seaborn as sns
from scipy.stats import ttest_ind, entropy, ks_2samp, chisquare, hypergeom, chi2_contingency, mannwhitneyu, anderson_ksamp, fisher_exact
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Define the path for the results directory (will be overridden in main function)
# This is just a placeholder - the actual directory is created in main()
results_dir = None

##############################################
# Enhanced Analysis Functions
##############################################

def analyze_gene_set_enrichment_enhanced(gene_list, baseline_genes, gene_set_name, evolutionary_order):
    """Enhanced enrichment analysis with both over and under-representation tests."""
    
    filtered_genes = baseline_genes[baseline_genes[0].isin(gene_list)]
    
    if len(filtered_genes) == 0:
        print(f"No genes found for {gene_set_name}")
        return None, None
    
    observed_counts = filtered_genes[1].value_counts()
    observed_percentages = (observed_counts / observed_counts.sum()) * 100
    
    for category in evolutionary_order:
        if category not in observed_percentages:
            observed_percentages[category] = 0
    
    observed_percentages = observed_percentages.reindex(evolutionary_order)
    
    baseline_counts = baseline_genes[1].value_counts()
    baseline_percentages = (baseline_counts / baseline_counts.sum()) * 100
    
    for category in evolutionary_order:
        if category not in baseline_percentages:
            baseline_percentages[category] = 0
    
    baseline_percentages = baseline_percentages.reindex(evolutionary_order)
    
    results = []
    total_observed = observed_counts.sum()
    total_baseline = baseline_genes.shape[0]
    
    for category in evolutionary_order:
        observed = observed_counts.get(category, 0)
        expected = baseline_genes[baseline_genes[1] == category].shape[0]
        
        if expected > 0 and total_baseline > 0:
            try:
                # Over-representation test
                p_value_over = hypergeom.sf(observed - 1, total_baseline, expected, total_observed)
                # Under-representation test
                p_value_under = hypergeom.cdf(observed, total_baseline, expected, total_observed)
            except:
                p_value_over = 1.0
                p_value_under = 1.0
        else:
            p_value_over = 1.0
            p_value_under = 1.0
        
        fold_change = (observed / total_observed) / (expected / total_baseline) if expected > 0 else 0
        enrichment_ratio = observed_percentages[category] / baseline_percentages[category] if baseline_percentages[category] > 0 else 0
        
        chi2, chi2_p, cramers_v = safe_chi2_contingency(observed, expected, total_observed, total_baseline)
        
        # Determine if over or under-represented
        if fold_change > 1.1:
            representation_type = "Over-represented"
            p_value_rep = p_value_over
        elif fold_change < 0.9:
            representation_type = "Under-represented"
            p_value_rep = p_value_under
        else:
            representation_type = "No change"
            p_value_rep = 1.0
        
        results.append({
            'Category': category,
            'Observed_Count': observed,
            'Expected_Count': expected,
            'Observed_Percentage': observed_percentages[category],
            'Baseline_Percentage': baseline_percentages[category],
            'Enrichment_Ratio': enrichment_ratio,
            'Fold_Change': fold_change,
            'P_Value_Over': p_value_over,
            'P_Value_Under': p_value_under,
            'P_Value_Representation': p_value_rep,
            'Representation_Type': representation_type,
            'Chi2_P_Value': chi2_p,
            'Cramers_V': cramers_v,
            'Significant_Over': p_value_over <= 0.05,
            'Significant_Under': p_value_under <= 0.05,
            'Significant_Rep': p_value_rep <= 0.05
        })
    
    results_df = pd.DataFrame(results)
    
    try:
        results_df['Bonferroni_P'] = multipletests(results_df['P_Value_Representation'], method='bonferroni')[1]
        results_df['FDR_P'] = multipletests(results_df['P_Value_Representation'], method='fdr_bh')[1]
        results_df['Significant_Bonferroni'] = results_df['Bonferroni_P'] <= 0.05
        results_df['Significant_FDR'] = results_df['FDR_P'] <= 0.05
    except Exception as e:
        print(f"Warning: Multiple comparison correction failed: {e}")
        results_df['Bonferroni_P'] = results_df['P_Value_Representation']
        results_df['FDR_P'] = results_df['P_Value_Representation']
        results_df['Significant_Bonferroni'] = results_df['Significant_Rep']
        results_df['Significant_FDR'] = results_df['Significant_Rep']
    
    gene_ages = filtered_genes[1].tolist()
    baseline_ages = baseline_genes[1].tolist()
    
    cumulative_p_values, cumulative_enrichment = calculate_cumulative_hypergeometric_test(
        gene_ages, baseline_ages, evolutionary_order
    )
    
    # Calculate percentage distributions for KS test
    gene_counts = pd.Series(gene_ages).value_counts()
    baseline_counts = pd.Series(baseline_ages).value_counts()
    
    # Ensure all categories are present
    for category in evolutionary_order:
        if category not in gene_counts:
            gene_counts[category] = 0
        if category not in baseline_counts:
            baseline_counts[category] = 0
    
    # Reindex to match evolutionary order
    gene_counts = gene_counts.reindex(evolutionary_order, fill_value=0)
    baseline_counts = baseline_counts.reindex(evolutionary_order, fill_value=0)
    
    # Convert to percentages
    gene_pct = (gene_counts / gene_counts.sum()) * 100
    baseline_pct = (baseline_counts / baseline_counts.sum()) * 100
    
    if len(gene_ages) > 0 and len(baseline_ages) > 0:
        try:
            # Create continuous distributions for statistical tests by repeating values according to their frequencies
            gene_continuous = []
            baseline_continuous = []
            
            for i, category in enumerate(evolutionary_order):
                # Repeat each category index according to its percentage
                gene_freq = int(round(gene_pct.iloc[i] * 100))  # Scale up for more data points
                baseline_freq = int(round(baseline_pct.iloc[i] * 100))
                
                gene_continuous.extend([i] * gene_freq)
                baseline_continuous.extend([i] * baseline_freq)
            
            # Calculate cumulative distributions
            gene_cdf = np.cumsum(gene_pct.values)
            baseline_cdf = np.cumsum(baseline_pct.values)
            
            # 1. CUMULATIVE HYPERGEOMETRIC TEST
            # Replicate individual test logic but for cumulative distributions
            # Test each cumulative category for over/under-representation
            cumulative_hypergeom_pvalues_over = []
            cumulative_hypergeom_pvalues_under = []
            cumulative_hypergeom_stats = []
            
            for i in range(len(evolutionary_order)):
                # Get actual cumulative counts (not percentages)
                gene_cumulative = int(round(gene_cdf[i] * len(gene_ages) / 100))
                baseline_cumulative = int(round(baseline_cdf[i] * len(baseline_ages) / 100))
                
                # Test if our cumulative count is significantly different from expected
                # This tests: "Are genes from this age AND ALL OLDER ages over/under-represented?"
                if gene_cumulative > 0 and baseline_cumulative > 0:
                    # Total genes in our set
                    total_genes = len(gene_ages)
                    # Total genes in baseline
                    total_baseline = len(baseline_ages)
                    
                    # Calculate expected cumulative count in our set based on baseline proportion
                    expected_cumulative = int(round((baseline_cumulative / total_baseline) * total_genes))
                    
                    # Hypergeometric test parameters: (k-1, N, K, n)
                    # k = observed cumulative count
                    # N = total population (baseline)
                    # K = expected cumulative count in baseline
                    # n = sample size (our gene set)
                    
                    # Over-representation test: is observed cumulative significantly higher than expected?
                    p_value_over = hypergeom.sf(gene_cumulative - 1, total_baseline, 
                                              baseline_cumulative, total_genes)
                    # Under-representation test: is observed cumulative significantly lower than expected?
                    p_value_under = hypergeom.cdf(gene_cumulative, total_baseline, 
                                                baseline_cumulative, total_genes)
                    
                    cumulative_hypergeom_pvalues_over.append(p_value_over)
                    cumulative_hypergeom_pvalues_under.append(p_value_under)
                    cumulative_hypergeom_stats.append(gene_cumulative - expected_cumulative)
                else:
                    cumulative_hypergeom_pvalues_over.append(1.0)
                    cumulative_hypergeom_pvalues_under.append(1.0)
                    cumulative_hypergeom_stats.append(0.0)
            
            # FDR correction for cumulative hypergeometric tests (over-representation)
            if cumulative_hypergeom_pvalues_over:
                cumulative_hypergeom_fdr_over = multipletests(cumulative_hypergeom_pvalues_over, method='fdr_bh')[1]
                cumulative_hypergeom_significant_over = np.sum(cumulative_hypergeom_fdr_over < 0.05)
            else:
                cumulative_hypergeom_fdr_over = np.array([1.0] * len(evolutionary_order))
                cumulative_hypergeom_significant_over = 0
            
            # FDR correction for cumulative hypergeometric tests (under-representation)
            if cumulative_hypergeom_pvalues_under:
                cumulative_hypergeom_fdr_under = multipletests(cumulative_hypergeom_pvalues_under, method='fdr_bh')[1]
                cumulative_hypergeom_significant_under = np.sum(cumulative_hypergeom_fdr_under < 0.05)
            else:
                cumulative_hypergeom_fdr_under = np.array([1.0] * len(evolutionary_order))
                cumulative_hypergeom_significant_under = 0
            
            # Total significant cumulative tests
            cumulative_hypergeom_significant = cumulative_hypergeom_significant_over + cumulative_hypergeom_significant_under
            
            
            # 2. MANN-WHITNEY U TEST
            # Test if genes tend to be from older evolutionary ages overall
            try:
                gene_ages_numeric = [evolutionary_order.index(age) + 1 for age in gene_ages if age in evolutionary_order]
                baseline_ages_numeric = [evolutionary_order.index(age) + 1 for age in baseline_ages if age in evolutionary_order]
                
                if len(gene_ages_numeric) > 0 and len(baseline_ages_numeric) > 0:
                    mw_stat, mw_pvalue = mannwhitneyu(gene_ages_numeric, baseline_ages_numeric, alternative='two-sided')
                else:
                    mw_stat, mw_pvalue = 0.0, 1.0
            except:
                mw_stat, mw_pvalue = 0.0, 1.0
            
            # 3. KOLMOGOROV-SMIRNOV TEST
            # Test if cumulative distributions are significantly different
            try:
                n_samples = 1000
                random_vals = np.random.uniform(0, 1, n_samples)
                gene_samples = np.interp(random_vals, gene_cdf/100, np.arange(len(gene_cdf)))
                baseline_samples = np.interp(random_vals, baseline_cdf/100, np.arange(len(baseline_cdf)))
                ks_stat, ks_pvalue = ks_2samp(gene_samples, baseline_samples)
            except:
                ks_stat, ks_pvalue = 0.0, 1.0
            # Use Mann-Whitney as the primary test for overall distribution comparison
            primary_stat, primary_pvalue = mw_stat, mw_pvalue
            
            
            # CDFs already calculated above

             
            
            area_between_curves = np.trapz(np.abs(gene_cdf - baseline_cdf))
            
            # Calculate mean age shift using weighted percentages
            gene_ages_numeric = [evolutionary_order.index(age) + 1 for age in gene_ages if age in evolutionary_order]
            baseline_ages_numeric = [evolutionary_order.index(age) + 1 for age in baseline_ages if age in evolutionary_order]
            mean_age_shift = np.mean(gene_ages_numeric) - np.mean(baseline_ages_numeric) if len(gene_ages_numeric) > 0 and len(baseline_ages_numeric) > 0 else 0
            
            continuous_results = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pvalue,
                'primary_statistic': primary_stat,
                'primary_pvalue': primary_pvalue,
                'cumulative_hypergeom_pvalues_over': cumulative_hypergeom_pvalues_over,
                'cumulative_hypergeom_pvalues_under': cumulative_hypergeom_pvalues_under,
                'cumulative_hypergeom_fdr_over': cumulative_hypergeom_fdr_over,
                'cumulative_hypergeom_fdr_under': cumulative_hypergeom_fdr_under,
                'cumulative_hypergeom_significant_over': cumulative_hypergeom_significant_over,
                'cumulative_hypergeom_significant_under': cumulative_hypergeom_significant_under,
                'cumulative_hypergeom_significant': cumulative_hypergeom_significant,
                'area_between_curves': area_between_curves,
                'mean_age_shift': mean_age_shift,
                'cumulative_p_values': cumulative_p_values,
                'cumulative_enrichment': cumulative_enrichment,
                'gene_cdf': gene_cdf,
                'baseline_cdf': baseline_cdf
            }
        except Exception as e:
            print(f"Warning: Continuous analysis failed: {e}")
            continuous_results = None
    else:
        continuous_results = None
    
    return results_df, continuous_results

def plot_enhanced_enrichment_analysis(results_df, continuous_results, gene_set_name, output_path, upregulated_genes=None, downregulated_genes=None, age_human_genes=None):
    """Create enhanced enrichment plots with raw counts, percentages, and representation types."""
    
    evolutionary_order = results_df['Category'].tolist()
    
    fig, axes = plt.subplots(3, 3, figsize=(32, 28))
    fig.suptitle(f'Enhanced Enrichment Analysis: {gene_set_name}', fontsize=32, fontweight='bold', y=0.95)
    
    # Plot 1: Stacked raw counts (if gene lists provided) or Raw counts comparison
    ax1 = axes[0, 0]
    x = range(len(evolutionary_order))
    
    if upregulated_genes is not None and downregulated_genes is not None and age_human_genes is not None:
        # Create stacked raw counts histogram with expected comparison
        # Filter genes
        upregulated_filtered = age_human_genes[age_human_genes[0].isin(upregulated_genes)]
        downregulated_filtered = age_human_genes[age_human_genes[0].isin(downregulated_genes)]
        
        upregulated_counts = upregulated_filtered[1].value_counts()
        downregulated_counts = downregulated_filtered[1].value_counts()
        
        # Get baseline distribution
        baseline_counts = age_human_genes[1].value_counts()
        
        # Ensure all categories are present
        for category in evolutionary_order:
            if category not in upregulated_counts:
                upregulated_counts[category] = 0
            if category not in downregulated_counts:
                downregulated_counts[category] = 0
            if category not in baseline_counts:
                baseline_counts[category] = 0
        
        # Reindex to match evolutionary order
        upregulated_counts = upregulated_counts.reindex(evolutionary_order)
        downregulated_counts = downregulated_counts.reindex(evolutionary_order)
        baseline_counts = baseline_counts.reindex(evolutionary_order)
        
        # Scale baseline to match total observed genes for fair comparison
        total_observed = upregulated_counts.sum() + downregulated_counts.sum()
        total_baseline = baseline_counts.sum()
        scaled_baseline = (baseline_counts / total_baseline) * total_observed
        
        # Create stacked bars for raw counts
        width = 0.35
        bars_up = ax1.bar([i - width/2 for i in x], upregulated_counts.values, 
                         width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
        bars_down = ax1.bar([i - width/2 for i in x], downregulated_counts.values, 
                           width, bottom=upregulated_counts.values,  # Stack on top of upregulated
                           label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
        
        # Add expected baseline bar
        bars_expected = ax1.bar([i + width/2 for i in x], scaled_baseline.values, 
                               width, label='Expected\n(Litman et al.)', alpha=0.7, color='#2ca02c')  # Green
        
        ax1.set_xlabel('Evolutionary Age')
        ax1.set_ylabel('Gene Count')
        ax1.set_title('Stacked Raw Counts: Observed vs Expected')
        ax1.set_xticks(x)
        ax1.set_xticklabels(evolutionary_order, rotation=90, ha='center')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add significance stars for raw counts using enrichment results
        # We need to get the enrichment results for the combined analysis
        combined_genes = list(upregulated_genes) + list(downregulated_genes)
        combined_results, _ = analyze_gene_set_enrichment_enhanced(combined_genes, age_human_genes, "Combined", evolutionary_order)
        
        if combined_results is not None:
            for i, category in enumerate(evolutionary_order):
                category_result = combined_results[combined_results['Category'] == category]
                if not category_result.empty:
                    p_value = category_result['P_Value_Representation'].iloc[0]
                    
                    # Add significance stars using FDR-corrected p-values
                    fdr_significant = category_result['Significant_FDR'].iloc[0]
                    if fdr_significant:
                        star_color = 'red' if category_result['Significant_Over'].iloc[0] else 'black'
                        ax1.text(i - width/2, upregulated_counts.values[i] + downregulated_counts.values[i] + 15, '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
        

    
    else:
        # Original raw counts comparison
        width = 0.35
        ax1.bar([i - width/2 for i in x], results_df['Observed_Count'], 
                width, label='Observed', alpha=0.7, color='#1f77b4')  # Blue
        ax1.bar([i + width/2 for i in x], results_df['Expected_Count'], 
                width, label='Expected\n(Litman et al.)', alpha=0.7, color='#ff7f0e')  # Orange
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            if row['Significant_FDR']:
                star_color = 'red' if row['Significant_Over'] else 'black'
                ax1.text(i, max(row['Observed_Count'], row['Expected_Count']) + 10, 
                        '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
        
        ax1.set_xlabel('Evolutionary Age')
        ax1.set_ylabel('Gene Count')
        ax1.set_title('Raw Counts: Observed vs Expected')
        ax1.set_xticks(x)
        ax1.set_xticklabels(evolutionary_order, rotation=90, ha='center')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stacked percentages (if gene lists provided) or Percentages comparison
    ax2 = axes[0, 1]
    
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
        baseline_pct = (baseline_counts / baseline_counts.sum()) * 100
        
        # Create stacked bars for percentages
        width = 0.35
        bars_up_pct = ax2.bar([i - width/2 for i in x], upregulated_pct.values, 
                             width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
        bars_down_pct = ax2.bar([i - width/2 for i in x], downregulated_pct.values, 
                               width, bottom=upregulated_pct.values,  # Stack on top of upregulated
                               label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
        
        # Add expected baseline bar
        bars_expected_pct = ax2.bar([i + width/2 for i in x], baseline_pct.values, 
                                   width, label='Expected\n(Litman et al.)', alpha=0.7, color='#2ca02c')  # Green
        
        ax2.set_xlabel('Evolutionary Age')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Stacked Percentages: Observed vs Expected')
        ax2.set_xticks(x)
        ax2.set_xticklabels(evolutionary_order, rotation=90, ha='center')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add significance stars for percentages using enrichment results
        if combined_results is not None:
            for i, category in enumerate(evolutionary_order):
                category_result = combined_results[combined_results['Category'] == category]
                if not category_result.empty:
                    p_value = category_result['P_Value_Representation'].iloc[0]
                    
                    # Add significance stars using FDR-corrected p-values
                    fdr_significant = category_result['Significant_FDR'].iloc[0]
                    if fdr_significant:
                        star_color = 'red' if category_result['Significant_Over'].iloc[0] else 'black'
                        ax2.text(i - width/2, upregulated_pct.values[i] + downregulated_pct.values[i] + 2, '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
    
    else:
        # Original percentages comparison
        width = 0.35
        ax2.bar([i - width/2 for i in x], results_df['Observed_Percentage'], 
                width, label='Observed', alpha=0.7, color='#1f77b4')  # Blue
        ax2.bar([i + width/2 for i in x], results_df['Baseline_Percentage'], 
                width, label='Expected\n(Litman et al.)', alpha=0.7, color='#ff7f0e')  # Orange
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            if row['Significant_FDR']:
                star_color = 'red' if row['Significant_Over'] else 'black'
                ax2.text(i, max(row['Observed_Percentage'], row['Baseline_Percentage']) + 0.5, 
                        '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
        
        ax2.set_xlabel('Evolutionary Age')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Percentages: Observed vs Expected')
        ax2.set_xticks(x)
        ax2.set_xticklabels(evolutionary_order, rotation=90, ha='center')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fold changes with representation types
    ax3 = axes[0, 2]
    colors = []
    for _, row in results_df.iterrows():
        if row['Representation_Type'] == 'Over-represented':
            colors.append('#2ca02c')  # Green for over-represented
        elif row['Representation_Type'] == 'Under-represented':
            colors.append('#d62728')  # Red for under-represented
        else:
            colors.append('#7f7f7f')  # Gray for no change
    
    bars = ax3.bar(x, results_df['Fold_Change'], color=colors, alpha=0.7)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        if row['Significant_FDR']:
            star_color = 'red' if row['Significant_Over'] else 'black'
            ax3.text(i, row['Fold_Change'] + 0.05, '*', 
                    ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
    
    ax3.set_xlabel('Evolutionary Age')
    ax3.set_ylabel('Fold Change')
    ax3.set_title('Fold Change vs Baseline (Red=Over, Blue=Under)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(evolutionary_order, rotation=90, ha='center')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Over-representation p-values
    ax4 = axes[1, 0]
    neg_log_pvals = -np.log10(results_df['P_Value_Over'])
    colors = ['#9467bd' if p <= 0.05 else '#c5c5c5' for p in results_df['P_Value_Over']]  # Purple for significant
    ax4.bar(x, neg_log_pvals, color=colors, alpha=0.7)
    
    ax4.axhline(y=-np.log10(0.05), color='#9467bd', linestyle='--', alpha=0.7, label='p=0.05')
    ax4.axhline(y=-np.log10(0.05/len(evolutionary_order)), color='#ff7f0e', linestyle='--', alpha=0.7, label='Bonferroni')
    
    ax4.set_xlabel('Evolutionary Age')
    ax4.set_ylabel('-log10(P-value)')
    ax4.set_title('Over-representation Significance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(evolutionary_order, rotation=90, ha='center')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Under-representation p-values
    ax5 = axes[1, 1]
    neg_log_pvals = -np.log10(results_df['P_Value_Under'])
    colors = ['#8c564b' if p <= 0.05 else '#c5c5c5' for p in results_df['P_Value_Under']]  # Brown for significant
    ax5.bar(x, neg_log_pvals, color=colors, alpha=0.7)
    
    ax5.axhline(y=-np.log10(0.05), color='#8c564b', linestyle='--', alpha=0.7, label='p=0.05')
    ax5.axhline(y=-np.log10(0.05/len(evolutionary_order)), color='#ff7f0e', linestyle='--', alpha=0.7, label='Bonferroni')
    
    ax5.set_xlabel('Evolutionary Age')
    ax5.set_ylabel('-log10(P-value)')
    ax5.set_title('Under-representation Significance')
    ax5.set_xticks(x)
    ax5.set_xticklabels(evolutionary_order, rotation=90, ha='center')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Effect sizes (Cramer's V)
    ax6 = axes[1, 2]
    colors = ['#e377c2' if v > 0.1 else '#17becf' for v in results_df['Cramers_V']]  # Pink for large, cyan for small
    ax6.bar(x, results_df['Cramers_V'], color=colors, alpha=0.7)
    ax6.axhline(y=0.1, color='#e377c2', linestyle='--', alpha=0.7, label='Small effect')
    ax6.axhline(y=0.3, color='#ff7f0e', linestyle='--', alpha=0.7, label='Medium effect')
    
    ax6.set_xlabel('Evolutionary Age')
    ax6.set_ylabel("Cramer's V")
    ax6.set_title('Effect Size')
    ax6.set_xticks(x)
    ax6.set_xticklabels(evolutionary_order, rotation=90, ha='center')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Representation type summary
    ax7 = axes[2, 0]
    rep_counts = results_df['Representation_Type'].value_counts()
    colors = ['#2ca02c' if 'Over' in rep else '#d62728' if 'Under' in rep else '#7f7f7f' for rep in rep_counts.index]
    ax7.pie(rep_counts.values, labels=rep_counts.index, autopct='%1.1f%%', colors=colors)
    ax7.set_title('Representation Type Distribution')
    
    # Plot 8: Continuous analysis
    ax8 = axes[2, 1]
    if continuous_results:
        x_cont = range(len(evolutionary_order))
        ax8.plot(x_cont, continuous_results['gene_cdf'], color='#1f77b4', linewidth=2, label='Gene Set', marker='o')
        ax8.plot(x_cont, continuous_results['baseline_cdf'], color='#ff7f0e', linewidth=2, label='Baseline', marker='s')
        
        ax8.set_xlabel('Evolutionary Age')
        ax8.set_ylabel('Cumulative Percentage (%)')
        ax8.set_title(f'Cumulative Distribution\nKS p={continuous_results["ks_pvalue"]:.2e}')
        ax8.set_xticks(x_cont)
        ax8.set_xticklabels(evolutionary_order, rotation=90, ha='center')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No continuous data available', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Continuous Analysis')
    
    # Plot 9: Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    total_genes = results_df['Observed_Count'].sum()
    significant_over = results_df['Significant_Over'].sum()
    significant_under = results_df['Significant_Under'].sum()
    significant_fdr = results_df['Significant_FDR'].sum()
    
    summary_text = f"""
    Summary Statistics:
    
    Total Genes: {total_genes}
    Significant Over-represented: {significant_over}
    Significant Under-represented: {significant_under}
    FDR Significant: {significant_fdr}
    
    Most Enriched: {results_df.loc[results_df['Fold_Change'].idxmax(), 'Category']}
    Fold Change: {results_df['Fold_Change'].max():.2f}x
    
    Most Depleted: {results_df.loc[results_df['Fold_Change'].idxmin(), 'Category']}
    Fold Change: {results_df['Fold_Change'].min():.2f}x
    """
    
    if continuous_results:
        summary_text += f"""
    
    Continuous Analysis:
    KS p-value: {continuous_results['ks_pvalue']:.2e}
    Mean age shift: {continuous_results['mean_age_shift']:.2f}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=40, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()
    
    return fig

def plot_stacked_histogram_enhanced(upregulated, downregulated, age_human_genes, evolutionary_order, output_path):
    """Create enhanced stacked histogram with raw counts and percentages."""
    
    # Filter genes
    upregulated_filtered = age_human_genes[age_human_genes[0].isin(upregulated)]
    downregulated_filtered = age_human_genes[age_human_genes[0].isin(downregulated)]
    
    upregulated_counts = upregulated_filtered[1].value_counts()
    downregulated_counts = downregulated_filtered[1].value_counts()
    
    # Ensure all categories are present
    for category in evolutionary_order:
        if category not in upregulated_counts:
            upregulated_counts[category] = 0
        if category not in downregulated_counts:
            downregulated_counts[category] = 0
    
    # Reindex to match evolutionary order
    upregulated_counts = upregulated_counts.reindex(evolutionary_order)
    downregulated_counts = downregulated_counts.reindex(evolutionary_order)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(28, 16))
    
    x = range(len(evolutionary_order))
    width = 0.35
    
    # Plot 1: Raw counts (side by side)
    bars1 = ax1.bar([i - width/2 for i in x], upregulated_counts.values, 
                    width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
    bars2 = ax1.bar([i + width/2 for i in x], downregulated_counts.values, 
                    width, label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
    
    ax1.set_xlabel('Evolutionary Age', fontsize=32, fontweight='bold')
    ax1.set_ylabel('Gene Count', fontsize=32, fontweight='bold')
    ax1.set_title("Raw Counts: Upregulated vs Downregulated Genes", fontsize=36, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=36)
    ax1.legend(fontsize=40)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=32)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=32)
    
    # Plot 2: Percentages (side by side)
    total_up = upregulated_counts.sum()
    total_down = downregulated_counts.sum()
    
    upregulated_pct = (upregulated_counts / total_up) * 100
    downregulated_pct = (downregulated_counts / total_down) * 100
    
    bars3 = ax2.bar([i - width/2 for i in x], upregulated_pct.values, 
                    width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
    bars4 = ax2.bar([i + width/2 for i in x], downregulated_pct.values, 
                    width, label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
    
    ax2.set_xlabel('Evolutionary Age', fontsize=32, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=40, fontweight='bold')
    ax2.set_title("Percentages: Upregulated vs Downregulated Genes", fontsize=36, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=36)
    ax2.legend(fontsize=40)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=32)
    
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=32)
    
    # Plot 3: STACKED histogram showing proportion of up/down in each category
    # Calculate total genes per category
    total_combined = upregulated_counts + downregulated_counts
    
    # Calculate proportions
    up_proportion = (upregulated_counts / total_combined) * 100
    down_proportion = (downregulated_counts / total_combined) * 100
    
    # Create stacked bars
    bars5 = ax3.bar(x, up_proportion.values, 
                    label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
    bars6 = ax3.bar(x, down_proportion.values, 
                    bottom=up_proportion.values,  # Stack on top of upregulated
                    label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
    
    ax3.set_xlabel('Evolutionary Age', fontsize=32, fontweight='bold')
    ax3.set_ylabel('Proportion (%)', fontsize=40, fontweight='bold')
    ax3.set_title("Stacked Proportions: Upregulated vs Downregulated Genes per Category", fontsize=36, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=36)
    ax3.legend(fontsize=40)
    ax3.grid(True, alpha=0.3)
    
    # Add proportion labels on stacked bars
    for i, (up_pct, down_pct, total_count) in enumerate(zip(up_proportion.values, down_proportion.values, total_combined.values)):
        if total_count > 0:
            # Label for upregulated portion
            if up_pct > 5:  # Only label if proportion is > 5%
                ax3.text(i, up_pct/2, f'{up_pct:.1f}%', ha='center', va='center', fontsize=44, fontweight='bold')
            
            # Label for downregulated portion
            if down_pct > 5:  # Only label if proportion is > 5%
                ax3.text(i, up_pct + down_pct/2, f'{down_pct:.1f}%', ha='center', va='center', fontsize=44, fontweight='bold')
            
            # Total count label at the top
            ax3.text(i, up_pct + down_pct + 2, f'n={int(total_count)}', ha='center', va='bottom', fontsize=40, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()
    
    return fig

def plot_single_stacked_histogram(upregulated, downregulated, age_human_genes, evolutionary_order, output_path, results_df=None):
    """Create SINGLE stacked histogram with up/down vs expected baseline."""
    
    # Filter genes
    upregulated_filtered = age_human_genes[age_human_genes[0].isin(upregulated)]
    downregulated_filtered = age_human_genes[age_human_genes[0].isin(downregulated)]
    
    upregulated_counts = upregulated_filtered[1].value_counts()
    downregulated_counts = downregulated_filtered[1].value_counts()
    
    # Ensure all categories are present
    for category in evolutionary_order:
        if category not in upregulated_counts:
            upregulated_counts[category] = 0
        if category not in downregulated_counts:
            downregulated_counts[category] = 0
    
    # Reindex to match evolutionary order
    upregulated_counts = upregulated_counts.reindex(evolutionary_order)
    downregulated_counts = downregulated_counts.reindex(evolutionary_order)
    
    # Get baseline counts and convert to percentages
    baseline_counts = age_human_genes[1].value_counts()
    baseline_counts = baseline_counts.reindex(evolutionary_order, fill_value=0)
    baseline_pct = (baseline_counts / baseline_counts.sum()) * 100
    
    # Convert up/down counts to percentages
    total_combined = upregulated_counts.sum() + downregulated_counts.sum()
    upregulated_pct = (upregulated_counts / total_combined) * 100
    downregulated_pct = (downregulated_counts / total_combined) * 100
    
    # Create SINGLE figure
    fig, ax = plt.subplots(1, 1, figsize=(28, 16))
    
    x = range(len(evolutionary_order))
    width = 0.35
    
    # Create stacked bars for up/down vs expected (ALL PERCENTAGES)
    bars_up = ax.bar([i - width/2 for i in x], upregulated_pct.values, 
                     width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
    bars_down = ax.bar([i - width/2 for i in x], downregulated_pct.values, 
                       width, bottom=upregulated_pct.values,  # Stack on top
                       label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
    
    # Add expected baseline bar (PERCENTAGES)
    bars_expected = ax.bar([i + width/2 for i in x], baseline_pct.values, 
                           width, label='Expected\n(Litman et al.)', alpha=0.7, color='#2ca02c')  # Green
    
    ax.set_xlabel('Evolutionary Age', fontsize=36, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=40, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=32, fontweight='bold')
    ax.tick_params(axis='y', labelsize=32)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    legend = ax.legend(fontsize=40)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax.grid(True, alpha=0.3)
    
    # Add significance stars if results are provided
    if results_df is not None:
        for i, category in enumerate(evolutionary_order):
            category_result = results_df[results_df['Category'] == category]
            if not category_result.empty:
                p_value = category_result['P_Value_Representation'].iloc[0]
                is_over_represented = category_result['Significant_Over'].iloc[0]
                
                # Add significance stars using FDR-corrected p-values
                fdr_significant = category_result['Significant_FDR'].iloc[0]
                if fdr_significant:
                    star_color = 'red' if is_over_represented else 'black'
                    ax.text(i - width/2, upregulated_pct.values[i] + downregulated_pct.values[i] + 2, '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()
    
    return fig

def plot_individual_stacked_percentages(results_df, gene_set_name, output_path, upregulated_genes=None, downregulated_genes=None, age_human_genes=None):
    """Create individual figure for stacked percentages matching the second subfigure exactly."""
    
    evolutionary_order = results_df['Category'].tolist()
    
    fig, ax = plt.subplots(1, 1, figsize=(28, 16))
    
    if upregulated_genes is not None and downregulated_genes is not None and age_human_genes is not None:
        # EXACT COPY OF SUBFIGURE 2 CODE - NO CHANGES
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
        baseline_pct = (baseline_counts / baseline_counts.sum()) * 100
        baseline_pct = baseline_pct.reindex(evolutionary_order, fill_value=0)
        
        # Create stacked bars for percentages
        x = range(len(evolutionary_order))
        width = 0.35
        bars_up_pct = ax.bar([i - width/2 for i in x], upregulated_pct.values, 
                             width, label='Upregulated', alpha=0.7, color='#1f77b4')  # Blue
        bars_down_pct = ax.bar([i - width/2 for i in x], downregulated_pct.values, 
                               width, bottom=upregulated_pct.values,  # Stack on top of upregulated
                               label='Downregulated', alpha=0.7, color='#ff7f0e')  # Orange
        
        # Add expected baseline bar
        bars_expected_pct = ax.bar([i + width/2 for i in x], baseline_pct.values, 
                                   width, label='Expected\n(Litman et al.)', alpha=0.7, color='#2ca02c')  # Green
        
        ax.set_xlabel('Evolutionary Age', fontsize=36, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=40, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=32, fontweight='bold')
        ax.tick_params(axis='y', labelsize=32)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        legend = ax.legend(fontsize=40)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
        
        # Add significance stars for percentages using enrichment results
        for i, category in enumerate(evolutionary_order):
            category_result = results_df[results_df['Category'] == category]
            if not category_result.empty:
                p_value = category_result['P_Value_Representation'].iloc[0]
                is_over_represented = category_result['Significant_Over'].iloc[0]
                
                # Add significance stars using FDR-corrected p-values
                fdr_significant = category_result['Significant_FDR'].iloc[0]
                if fdr_significant:
                    star_color = 'red' if is_over_represented else 'black'
                    ax.text(i - width/2, upregulated_pct.values[i] + downregulated_pct.values[i] + 2, '*', ha='center', va='bottom', fontsize=48, fontweight='bold', color=star_color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()
    
    return fig

def plot_cumulative_hypergeometric_distribution(continuous_results, gene_set_name, output_path):
    """Create individual figure for cumulative hypergeometric distribution with mean age shift."""
    
    if not continuous_results:
        print(f"No continuous results available for {gene_set_name}")
        return None
    
    evolutionary_order = ['All living organisms', 'Eukaryota', 'Opisthokonta', 'Holozoa', 'Metazoa', 
                         'Eumetazoa', 'Bilateria', 'Deuterostomia', 'Chordata', 'Olfactores', 
                         'Craniata', 'Euteleostomi', 'Tetrapoda', 'Amniota', 'Mammalia', 
                         'Eutheria', 'Boreoeutheria', 'Euarchontoglires', 'Primates']
    
    fig, ax = plt.subplots(1, 1, figsize=(28, 16))
    
    x = range(len(evolutionary_order))
    
    # Plot cumulative distributions
    ax.plot(x, continuous_results['gene_cdf'], color='#1f77b4', linewidth=3, 
            label='Gene Set', marker='o', markersize=6, alpha=0.8)
    ax.plot(x, continuous_results['baseline_cdf'], color='#ff7f0e', linewidth=3, 
            label='Baseline (Litman et al.)', marker='s', markersize=6, alpha=0.8)
    
    # Fill area between curves
    ax.fill_between(x, continuous_results['gene_cdf'], continuous_results['baseline_cdf'], 
                    alpha=0.3, color='gray', label='Area between curves')
    
    # Note: Stars removed from cumulative distribution figures as they were showing too many significant results
    # The Mann-Whitney U test provides the overall statistical significance for the cumulative distribution difference
    
    # Add Mann-Whitney U test only
    mw_pval = continuous_results['mw_pvalue']
    pval_text = f"Mann-Whitney U p-value: {mw_pval:.2e}"
    if mw_pval < 0.05:
        pval_text += ' (*)'
    
    # Add mean shift under statistical tests
    mean_shift = continuous_results['mean_age_shift']
    shift_text = f'Mean Age Shift: {mean_shift:.2f} categories'
    
    combined_text = f'{pval_text}\n{shift_text}'
    
    ax.text(0.02, 0.88, combined_text, transform=ax.transAxes, 
            fontsize=32, fontweight='bold')
    
    ax.set_xlabel('Evolutionary Age', fontsize=36, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=40, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(evolutionary_order, rotation=90, ha='center', fontsize=32, fontweight='bold')
    ax.tick_params(axis='y', labelsize=32)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    legend = ax.legend(fontsize=40, loc='lower right')
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()
    
    return fig

# Import helper functions from previous version
def calculate_cumulative_hypergeometric_test(gene_ages, baseline_ages, evolutionary_order):
    """Perform cumulative hypergeometric test across evolutionary ages."""
    
    age_mapping = {cat: i for i, cat in enumerate(evolutionary_order)}
    gene_ages_numeric = [age_mapping[age] for age in gene_ages if age in age_mapping]
    baseline_ages_numeric = [age_mapping[age] for age in baseline_ages if age in age_mapping]
    
    if len(gene_ages_numeric) == 0:
        return None, None
    
    cumulative_p_values = []
    cumulative_enrichment = []
    
    total_genes = len(gene_ages_numeric)
    total_baseline = len(baseline_ages_numeric)
    
    for i in range(len(evolutionary_order)):
        observed_cumulative = sum(1 for age in gene_ages_numeric if age <= i)
        expected_cumulative = sum(1 for age in baseline_ages_numeric if age <= i)
        
        if expected_cumulative > 0 and total_baseline > 0:
            try:
                p_value = hypergeom.sf(observed_cumulative - 1, total_baseline, expected_cumulative, total_genes)
                enrichment = (observed_cumulative / total_genes) / (expected_cumulative / total_baseline) if expected_cumulative > 0 else 0
            except:
                p_value = 1.0
                enrichment = 0
        else:
            p_value = 1.0
            enrichment = 0
        
        cumulative_p_values.append(p_value)
        cumulative_enrichment.append(enrichment)
    
    return cumulative_p_values, cumulative_enrichment

def safe_chi2_contingency(observed, expected, total_observed, total_baseline):
    """Safely calculate chi-square contingency table with error handling."""
    try:
        contingency_table = [[observed, total_observed - observed], 
                           [expected, total_baseline - expected]]
        
        if any(cell == 0 for row in contingency_table for cell in row):
            return 0, 1.0, 0
        
        chi2, p_value, dof, expected_chi2 = chi2_contingency(contingency_table)
        
        n = total_observed + total_baseline
        min_dim = min(2, 2) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if chi2 > 0 else 0
        
        return chi2, p_value, cramers_v
        
    except Exception as e:
        print(f"Warning: Chi-square calculation failed: {e}")
        return 0, 1.0, 0

def save_cumulative_hypergeometric_results(continuous_results, gene_set_name, output_dir):
    """Save cumulative hypergeometric test results to CSV."""
    
    if not continuous_results or 'cumulative_hypergeom_pvalues' not in continuous_results:
        print(f"No cumulative hypergeometric results to save for {gene_set_name}")
        return
    
    # Create results DataFrame
    evolutionary_order = [
        "All living organisms", "Eukaryota", "Opisthokonta", "Holozoa", "Metazoa",
        "Eumetazoa", "Bilateria", "Deuterostomia", "Chordata", "Olfactores",
        "Craniata", "Euteleostomi", "Tetrapoda", "Amniota", "Mammalia",
        "Eutheria", "Boreoeutheria", "Euarchontoglires", "Primates"
    ]
    
    results_data = []
    for i, category in enumerate(evolutionary_order):
        if i < len(continuous_results['cumulative_hypergeom_pvalues_over']):
            results_data.append({
                'Evolutionary_Age': category,
                'Cumulative_P_Value_Over': continuous_results['cumulative_hypergeom_pvalues_over'][i],
                'Cumulative_P_Value_Under': continuous_results['cumulative_hypergeom_pvalues_under'][i],
                'FDR_Corrected_P_Value_Over': continuous_results['cumulative_hypergeom_fdr_over'][i],
                'FDR_Corrected_P_Value_Under': continuous_results['cumulative_hypergeom_fdr_under'][i],
                'Significant_Over': continuous_results['cumulative_hypergeom_fdr_over'][i] < 0.05,
                'Significant_Under': continuous_results['cumulative_hypergeom_fdr_under'][i] < 0.05,
                'Cumulative_Count_Difference': continuous_results['cumulative_hypergeom_stats'][i] if 'cumulative_hypergeom_stats' in continuous_results else 0
            })
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f'{gene_set_name}_cumulative_hypergeometric_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved cumulative hypergeometric results to: {output_path}")

def generate_enhanced_report(results_df, continuous_results, gene_set_name):
    """Generate an enhanced analysis report."""
    
    print(f"\n{'='*80}")
    print(f"ENHANCED ANALYSIS REPORT: {gene_set_name}")
    print(f"{'='*80}")
    
    total_genes = results_df['Observed_Count'].sum()
    significant_over = results_df['Significant_Over'].sum()
    significant_under = results_df['Significant_Under'].sum()
    significant_fdr = results_df['Significant_FDR'].sum()
    
    print(f"Total genes analyzed: {total_genes}")
    print(f"Significant over-represented: {significant_over}/{len(results_df)}")
    print(f"Significant under-represented: {significant_under}/{len(results_df)}")
    print(f"FDR-corrected significant: {significant_fdr}/{len(results_df)}")
    
    print(f"\nMost over-represented categories (fold change > 1.5):")
    over_rep = results_df[results_df['Fold_Change'] > 1.5].sort_values('Fold_Change', ascending=False)
    if not over_rep.empty:
        for _, row in over_rep.iterrows():
            significance = "***" if row['Significant_FDR'] else "*" if row['Significant_Over'] else ""
            print(f"  {row['Category']}: {row['Fold_Change']:.2f}x (p={row['P_Value_Over']:.3e}) {significance}")
    else:
        print("  None found")
    
    print(f"\nMost under-represented categories (fold change < 0.7):")
    under_rep = results_df[results_df['Fold_Change'] < 0.7].sort_values('Fold_Change')
    if not under_rep.empty:
        for _, row in under_rep.iterrows():
            significance = "***" if row['Significant_FDR'] else "*" if row['Significant_Under'] else ""
            print(f"  {row['Category']}: {row['Fold_Change']:.2f}x (p={row['P_Value_Under']:.3e}) {significance}")
    else:
        print("  None found")
    
    if continuous_results:
        print(f"\nContinuous analysis results:")
        print(f"  Mann-Whitney U test p-value: {continuous_results['mw_pvalue']:.3e}")
        print(f"  Kolmogorov-Smirnov test p-value: {continuous_results['ks_pvalue']:.3e}")
        print(f"  Cumulative hypergeometric tests: {continuous_results['cumulative_hypergeom_significant']} significant (FDR < 0.05)")
        print(f"    - Over-represented: {continuous_results['cumulative_hypergeom_significant_over']}")
        print(f"    - Under-represented: {continuous_results['cumulative_hypergeom_significant_under']}")
        print(f"  Area between curves: {continuous_results['area_between_curves']:.2f}")
        print(f"  Mean age shift: {continuous_results['mean_age_shift']:.2f} categories")
        
        if continuous_results['mean_age_shift'] < 0:
            print("  → Overall shift toward OLDER evolutionary ages")
        elif continuous_results['mean_age_shift'] > 0:
            print("  → Overall shift toward YOUNGER evolutionary ages")
        else:
            print("  → No significant overall age shift")
    
    print(f"\nRepresentation type summary:")
    rep_summary = results_df['Representation_Type'].value_counts()
    for rep_type, count in rep_summary.items():
        print(f"  {rep_type}: {count} categories")
    
    print(f"{'='*80}\n")

##############################################
# Main Analysis Section
##############################################

def analyze_dataset(dataset_name, upregulated_genes, downregulated_genes, age_human_genes, evolutionary_order, results_dir):
    """Analyze a specific dataset with enhanced features."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Create dataset-specific results directory
    dataset_results_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # Convert to pandas Series for consistency
    upregulated_genes = pd.Series(upregulated_genes)
    downregulated_genes = pd.Series(downregulated_genes)
    
    # Combine for analysis
    combined_genes = pd.concat([upregulated_genes, downregulated_genes])
    
    print(f"Loaded {dataset_name} signature: {len(upregulated_genes)} upregulated, {len(downregulated_genes)} downregulated, {len(combined_genes)} combined")
    
    # Analyze upregulated genes
    print(f"\n{'-'*60}")
    print(f"ANALYZING {dataset_name.upper()} UPREGULATED GENES")
    print(f"{'-'*60}")
    
    enhanced_results, enhanced_continuous = analyze_gene_set_enrichment_enhanced(
        upregulated_genes.tolist(), age_human_genes, f"{dataset_name} Upregulated", evolutionary_order
    )
    
    if enhanced_results is not None:
        plot_path = os.path.join(dataset_results_dir, f'{dataset_name}_upregulated_enhanced_analysis.png')
        plot_enhanced_enrichment_analysis(
            enhanced_results, enhanced_continuous, f"{dataset_name} Upregulated", plot_path
        )
        
        generate_enhanced_report(enhanced_results, enhanced_continuous, f"{dataset_name} Upregulated")
        
        # Save cumulative hypergeometric results
        save_cumulative_hypergeometric_results(enhanced_continuous, f"{dataset_name} Upregulated", dataset_results_dir)
        enhanced_results.to_csv(os.path.join(dataset_results_dir, f'{dataset_name}_upregulated_enhanced_results.csv'), index=False)
        
        # Save continuous analysis results
        if enhanced_continuous:
            continuous_df = pd.DataFrame([enhanced_continuous])
            continuous_df.to_csv(os.path.join(dataset_results_dir, f'{dataset_name}_upregulated_enhanced_continuous_results.csv'), index=False)
    
    # Analyze downregulated genes
    print(f"\n{'-'*60}")
    print(f"ANALYZING {dataset_name.upper()} DOWNREGULATED GENES")
    print(f"{'-'*60}")
    
    enhanced_results, enhanced_continuous = analyze_gene_set_enrichment_enhanced(
        downregulated_genes.tolist(), age_human_genes, f"{dataset_name} Downregulated", evolutionary_order
    )
    
    if enhanced_results is not None:
        plot_path = os.path.join(dataset_results_dir, f'{dataset_name}_downregulated_enhanced_analysis.png')
        plot_enhanced_enrichment_analysis(
            enhanced_results, enhanced_continuous, f"{dataset_name} Downregulated", plot_path
        )
        
        generate_enhanced_report(enhanced_results, enhanced_continuous, f"{dataset_name} Downregulated")
        
        # Save cumulative hypergeometric results
        save_cumulative_hypergeometric_results(enhanced_continuous, f"{dataset_name} Downregulated", dataset_results_dir)
        enhanced_results.to_csv(os.path.join(dataset_results_dir, f'{dataset_name}_downregulated_enhanced_results.csv'), index=False)
        
        # Save continuous analysis results
        if enhanced_continuous:
            continuous_df = pd.DataFrame([enhanced_continuous])
            continuous_df.to_csv(os.path.join(dataset_results_dir, f'{dataset_name}_downregulated_enhanced_continuous_results.csv'), index=False)
    
    # Analyze combined genes
    print(f"\n{'-'*60}")
    print(f"ANALYZING {dataset_name.upper()} COMBINED GENES (UP + DOWN)")
    print(f"{'-'*60}")
    
    enhanced_results, enhanced_continuous = analyze_gene_set_enrichment_enhanced(
        combined_genes.tolist(), age_human_genes, f"{dataset_name} Combined", evolutionary_order
    )
    
    if enhanced_results is not None:
        plot_path = os.path.join(dataset_results_dir, f'{dataset_name}_combined_enhanced_analysis.png')
        plot_enhanced_enrichment_analysis(
            enhanced_results, enhanced_continuous, f"{dataset_name} Combined", plot_path,
            upregulated_genes=upregulated_genes.tolist(), 
            downregulated_genes=downregulated_genes.tolist(),
            age_human_genes=age_human_genes
        )
        
        generate_enhanced_report(enhanced_results, enhanced_continuous, f"{dataset_name} Combined")
        
        # Save cumulative hypergeometric results
        save_cumulative_hypergeometric_results(enhanced_continuous, f"{dataset_name} Combined", dataset_results_dir)
        enhanced_results.to_csv(os.path.join(dataset_results_dir, f'{dataset_name}_combined_enhanced_results.csv'), index=False)
        
        # Save continuous analysis results
        if enhanced_continuous:
            continuous_df = pd.DataFrame([enhanced_continuous])
            continuous_df.to_csv(os.path.join(dataset_results_dir, f'{dataset_name}_combined_enhanced_continuous_results.csv'), index=False)
    
    # Create stacked histogram (SINGLE figure with up/down vs expected)
    print(f"\nCreating enhanced stacked histogram for {dataset_name}...")
    stacked_plot_path = os.path.join(dataset_results_dir, f'{dataset_name}_stacked_histogram_enhanced.png')
    plot_single_stacked_histogram(
        upregulated_genes.tolist(), downregulated_genes.tolist(), age_human_genes, 
        evolutionary_order, stacked_plot_path, enhanced_results
    )
    
    # Create individual stacked percentages figure (EXACT copy of second subfigure)
    print(f"Creating individual stacked percentages figure for {dataset_name}...")
    stacked_pct_path = os.path.join(dataset_results_dir, f"{dataset_name}_stack_histogram_observed_vs_expected.png")
    plot_individual_stacked_percentages(enhanced_results, dataset_name, stacked_pct_path, 
                                       upregulated_genes.tolist(), downregulated_genes.tolist(), age_human_genes)
    
    # Create cumulative hypergeometric distribution figure (using COMBINED results)
    print(f"Creating cumulative hypergeometric distribution figure for {dataset_name}...")
    print(f"enhanced_continuous is: {enhanced_continuous}")
    if enhanced_continuous is not None:
        cumulative_dist_path = os.path.join(dataset_results_dir, f"{dataset_name}_cumulative_hypergeometric_distribution.png")
        plot_cumulative_hypergeometric_distribution(enhanced_continuous, dataset_name, cumulative_dist_path)
        print(f"Hypergeometric distribution figure created: {cumulative_dist_path}")
    else:
        print(f"No continuous results available for {dataset_name} - skipping hypergeometric figure")
    
    print(f"{dataset_name} analysis complete! Results saved in {dataset_results_dir}/")

def create_mann_whitney_results_table(results_dir):
    """Create a comprehensive table with Mann-Whitney U test results and mean age shift."""
    
    # Define all datasets and their combinations (using actual directory names)
    datasets = [
        # CellAge and AgeMeta
        ("CellAge", "Upregulated"),
        ("CellAge", "Downregulated"), 
        ("CellAge", "Combined"),
        ("AgeMeta", "Upregulated"),
        ("AgeMeta", "Downregulated"),
        ("AgeMeta", "Combined"),
        
        # Tissue-specific aging
        ("Skin_40-69", "Upregulated"),
        ("Skin_40-69", "Downregulated"),
        ("Skin_40-69", "Combined"),
        ("Skin_70+", "Upregulated"),
        ("Skin_70+", "Downregulated"),
        ("Skin_70+", "Combined"),
        ("Ovary", "Upregulated"),
        ("Ovary", "Downregulated"),
        ("Ovary", "Combined"),
        ("Progenitors", "Upregulated"),
        ("Progenitors", "Downregulated"),
        ("Progenitors", "Combined"),
        
        # Senescent signatures
        ("Mesenchymal_senescent", "Upregulated"),
        ("Mesenchymal_senescent", "Downregulated"),
        ("Mesenchymal_senescent", "Combined"),
        ("CellAge_Senescence", "Upregulated"),
        ("CellAge_Senescence", "Downregulated"),
        ("CellAge_Senescence", "Combined"),
        
        # Brain regions
        ("Brain_Cortex", "Upregulated"),
        ("Brain_Cortex", "Downregulated"),
        ("Brain_Cortex", "Combined"),
        ("Brain_Hippocampus", "Upregulated"),
        ("Brain_Hippocampus", "Downregulated"),
        ("Brain_Hippocampus", "Combined"),
        ("Brain_Cerebellum", "Upregulated"),
        ("Brain_Cerebellum", "Downregulated"),
        ("Brain_Cerebellum", "Combined"),
        
        # Other datasets
        ("Mesenchymal", "Upregulated"),
        ("Mesenchymal", "Downregulated"),
        ("Mesenchymal", "Combined"),
        ("CD8T", "Upregulated"),
        ("CD8T", "Downregulated"),
        ("CD8T", "Combined"),
    ]
    
    results = []
    
    for dataset, analysis_type in datasets:
        # Construct the CSV path
        if dataset.startswith("Brain_"):
            brain_region = dataset.replace("Brain_", "")
            csv_path = os.path.join(results_dir, "Brain", f"Brain_{brain_region}", f"{dataset}_{analysis_type}_enhanced_continuous_results.csv")
        else:
            csv_path = os.path.join(results_dir, dataset, f"{dataset}_{analysis_type}_enhanced_continuous_results.csv")
        
        # Extract results
        mw_pvalue = None
        mean_age_shift = None
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Look for Mann-Whitney U test results
                if 'mw_pvalue' in df.columns:
                    mw_pvalue = df['mw_pvalue'].iloc[0] if len(df) > 0 else None
                elif 'primary_pvalue' in df.columns:
                    mw_pvalue = df['primary_pvalue'].iloc[0] if len(df) > 0 else None
                    
                if 'mean_age_shift' in df.columns:
                    mean_age_shift = df['mean_age_shift'].iloc[0] if len(df) > 0 else None
                    
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        
        # Determine significance
        if mw_pvalue is not None:
            if mw_pvalue < 0.001:
                significance = "***"
            elif mw_pvalue < 0.01:
                significance = "**"
            elif mw_pvalue < 0.05:
                significance = "*"
            else:
                significance = "ns"
        else:
            significance = "N/A"
        
        # Determine age shift direction
        if mean_age_shift is not None:
            if mean_age_shift < -0.5:
                shift_direction = "Older"
            elif mean_age_shift > 0.5:
                shift_direction = "Younger"
            else:
                shift_direction = "Neutral"
        else:
            shift_direction = "N/A"
        
        results.append({
            'Dataset': dataset,
            'Analysis_Type': analysis_type,
            'Mann_Whitney_pvalue': mw_pvalue,
            'Significance': significance,
            'Mean_Age_Shift': mean_age_shift,
            'Shift_Direction': shift_direction
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(results_dir, "Mann_Whitney_Results_Table.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nMann-Whitney U test results table saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*80)
    print("MANN-WHITNEY U TEST RESULTS AND MEAN AGE SHIFT SUMMARY")
    print("="*80)
    
    # Group by dataset for better readability
    for dataset in df_results['Dataset'].unique():
        print(f"\n{dataset}:")
        print("-" * 40)
        subset = df_results[df_results['Dataset'] == dataset]
        for _, row in subset.iterrows():
            pval_str = f"{row['Mann_Whitney_pvalue']:.3e}" if row['Mann_Whitney_pvalue'] is not None else "N/A"
            shift_str = f"{row['Mean_Age_Shift']:.2f}" if row['Mean_Age_Shift'] is not None else "N/A"
            print(f"  {row['Analysis_Type']:12} | p = {pval_str:>10} {row['Significance']:>3} | Shift = {shift_str:>6} ({row['Shift_Direction']})")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    significant_count = len(df_results[df_results['Significance'].isin(['*', '**', '***'])])
    total_count = len(df_results[df_results['Mann_Whitney_pvalue'].notna()])
    
    older_shift_count = len(df_results[df_results['Shift_Direction'] == 'Older'])
    younger_shift_count = len(df_results[df_results['Shift_Direction'] == 'Younger'])
    neutral_shift_count = len(df_results[df_results['Shift_Direction'] == 'Neutral'])
    
    print(f"Total analyses: {total_count}")
    if total_count > 0:
        print(f"Significant (p < 0.05): {significant_count} ({significant_count/total_count*100:.1f}%)")
    else:
        print(f"Significant (p < 0.05): {significant_count} (N/A - no data)")
    print(f"Shift toward older ages: {older_shift_count}")
    print(f"Shift toward younger ages: {younger_shift_count}")
    print(f"Neutral shift: {neutral_shift_count}")
    
    return df_results

def create_combined_results_table_figure(results_dir):
    """Create a visual table figure showing only combined results."""
    
    # Read the Mann-Whitney results table
    table_path = os.path.join(results_dir, "Mann_Whitney_Results_Table.csv")
    if not os.path.exists(table_path):
        print(f"Table not found at {table_path}")
        return None
    
    df = pd.read_csv(table_path)
    
    # Filter for combined results only
    combined_df = df[df['Analysis_Type'] == 'Combined'].copy()
    
    if len(combined_df) == 0:
        print("No combined results found")
        return None
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the table
    table_data = []
    headers = ['Dataset', 'Mann-Whitney p-value', 'Significance', 'Mean Age Shift']
    
    for _, row in combined_df.iterrows():
        pval_str = f"{row['Mann_Whitney_pvalue']:.3e}" if row['Mann_Whitney_pvalue'] is not None else "N/A"
        shift_str = f"{row['Mean_Age_Shift']:.2f}" if row['Mean_Age_Shift'] is not None else "N/A"
        
        # Replace CellAge with GenAge
        dataset_name = row['Dataset']
        if dataset_name == 'CellAge':
            dataset_name = 'GenAge'
        
        table_data.append([
            dataset_name,
            pval_str,
            row['Significance'],
            shift_str
        ])
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 3.0)
    
    # Color code with consistent row colors and accessibility
    for i in range(len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#2E7D32')  # Dark green
                cell.set_text_props(weight='bold', color='white', size=24)
            else:
                # Consistent row colors (alternating light/dark)
                if i % 2 == 0:
                    cell.set_facecolor('#F8F9FA')  # Very light gray
                else:
                    cell.set_facecolor('#FFFFFF')  # White
                
                # Get significance
                significance = table_data[i-1][2]
                
                # If significant, make everything bold; if not significant, make normal
                if significance in ['***', '**', '*']:
                    cell.set_text_props(weight='bold', color='#000000', size=20)  # Bold black for significant
                else:
                    cell.set_text_props(weight='normal', color='#666666', size=20)  # Gray for non-significant
    
    # Add title with better formatting
    plt.title('Mann-Whitney U Test Results - Combined Gene Sets\n(*** p<0.001, ** p<0.01, * p<0.05)', 
              fontsize=28, fontweight='bold', pad=30, color='#1B5E20')
    
    # Add caption for the paper
    caption = ("Table 1. Mann-Whitney U test results for combined gene sets across multiple aging datasets. "
               "P-values indicate significance of distribution differences between gene sets and baseline evolutionary age distribution. "
               "Mean age shift represents the average change in evolutionary age categories (negative values indicate shift toward older ages). "
               "Significance levels: *** p<0.001, ** p<0.01, * p<0.05. Bold text indicates significant results (p<0.05).")
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=16, style='italic', color='#424242', wrap=True)
    
    # Save the figure
    output_path = os.path.join(results_dir, "Combined_Results_Table.png")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for caption
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined results table figure saved to: {output_path}")
    return output_path

def main():
    """Main analysis function with enhanced features for multiple datasets."""
    
    print("Starting Enhanced Multi-Dataset Atavism Analysis...")
    
    # Define evolutionary order
    evolutionary_order = [
        "All living organisms", "Eukaryota", "Opisthokonta", "Holozoa", "Metazoa",
        "Eumetazoa", "Bilateria", "Deuterostomia", "Chordata", "Olfactores",
        "Craniata", "Euteleostomi", "Tetrapoda", "Amniota", "Mammalia",
        "Eutheria", "Boreoeutheria", "Euarchontoglires", "Primates"
    ]
    
    # Load baseline genome data
    print("Loading baseline genome data...")
    age_human_genes = pd.read_csv('./Database_age_human_genes/1-s2.0-S0093775418302264-mmc2.csv', 
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
    
    # Create main results directory
    results_dir = f"Results_multi_dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # ========================================================================
    # CELLAGE ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING CELLAGE DATASET")
        print("="*80)
        
        # Load cellAge signature data using YOUR exact code structure
        cellAge_upregulated = pd.read_csv('./Database_age_human_genes/over.all.csv', header=0, delimiter = ';')
        cellAge_upregulated = cellAge_upregulated['Gene']
        
        cellAge_downregulated = pd.read_csv('./Database_age_human_genes/under.all.csv', header=0, delimiter = ';')
        cellAge_downregulated = cellAge_downregulated['Gene']
        
        # Analyze CellAge dataset
        analyze_dataset("CellAge", cellAge_upregulated, cellAge_downregulated, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing CellAge signature: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # AGEMETA ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING AGEMETA DATASET")
        print("="*80)
        
        # Load the dataset from the uploaded file
        file_path = './database_age_human_genes/aging_diffexpr_data.csv'
        df = pd.read_csv(file_path, sep='\t')
        
        # Ensure the dataset has the necessary columns
        required_columns = ['entrez', 'genesymbol', 'Brain_pval', 'Human_pval', 'Muscle_pval', 'Liver_pval', 'All_pval']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"The column '{col}' is missing from the dataset")
        
        # Filter for human-specific genes with p-value < 0.05 across tissues
        def filter_genes_by_pval(df, pval_column):
            return df[df[pval_column] < 0.05]
        
        conditions = {
            'brain_genes': filter_genes_by_pval(df, 'Brain_adj_pval'),
            'human_genes': filter_genes_by_pval(df, 'Human_adj_pval'),
            'muscle_genes': filter_genes_by_pval(df, 'Muscle_adj_pval'),
            'liver_genes': filter_genes_by_pval(df, 'Liver_adj_pval'),
            'all_genes': filter_genes_by_pval(df, 'All_adj_pval'),
        }
        
        # Create AgeMeta subfolder for all CSV files
        ageMeta_results_dir = os.path.join(results_dir, "AgeMeta")
        os.makedirs(ageMeta_results_dir, exist_ok=True)
        
        # Save each condition to a separate CSV file in AgeMeta subfolder
        for condition_name, condition_data in conditions.items():
            condition_output_path = os.path.join(ageMeta_results_dir, f'{condition_name}.csv')
            condition_data.to_csv(condition_output_path, index=False)
            print(f"Condition '{condition_name}' saved to {condition_output_path}")
        
        # Combine all genes into one DataFrame and drop duplicates
        all_filtered_genes = pd.concat(conditions.values()).drop_duplicates()
        
        # Save the combined results to a CSV file in AgeMeta subfolder
        combined_output_path = os.path.join(ageMeta_results_dir, 'combined_filtered_human_genes.csv')
        all_filtered_genes.to_csv(combined_output_path, index=False)
        print(f"Combined filtered genes saved to {combined_output_path}")
        
        # Display the first few rows of the result
        print(all_filtered_genes.head())
        
        # Reload human_genes for further processing
        human_genes = conditions['human_genes']
        
        # load mapping
        mapping_mouse_human = pd.read_csv('././database_age_human_genes/hcop-1737733278553_649genes_sig.txt', delimiter = '\t' )
        mapping_mouse_human = mapping_mouse_human[['Primary symbol', 'Ortholog symbol']]
        dict_mapping_mouse_human = {k: list(v) for k,v in mapping_mouse_human.groupby('Primary symbol')['Ortholog symbol']}
        dict_mapping_mouse_human =  {k: v[0] for k,v in dict_mapping_mouse_human.items()}
        
        # Translate mouse gene symbols to human gene symbols in human_genes
        human_genes['genesymbol'] = human_genes['genesymbol'].apply(lambda x: dict_mapping_mouse_human.get(x, x))
        
        # Display the updated human_genes DataFrame
        print(human_genes.head())
        
        # Filter genes based on Human_logFC values
        human_genes_up = human_genes[human_genes['Human_logFC'] > 0]
        human_genes_down = human_genes[human_genes['Human_logFC'] < 0]
        
        # Save the filtered results to separate CSV files in AgeMeta subfolder
        human_genes_up_path = os.path.join(ageMeta_results_dir, 'human_genes_up.csv')
        human_genes_down_path = os.path.join(ageMeta_results_dir, 'human_genes_down.csv')
        
        human_genes_up.to_csv(human_genes_up_path, index=False)
        human_genes_down.to_csv(human_genes_down_path, index=False)
        
        # Display summary of the filtering
        print(f"Number of upregulated genes: {len(human_genes_up)}")
        print(f"Number of downregulated genes: {len(human_genes_down)}")
        print(f"Upregulated genes saved to: {human_genes_up_path}")
        print(f"Downregulated genes saved to: {human_genes_down_path}")
        
        # Convert to Series for analysis
        ageMeta_upregulated = human_genes_up['genesymbol']
        ageMeta_downregulated = human_genes_down['genesymbol']
        
        # Analyze AgeMeta dataset
        analyze_dataset("AgeMeta", ageMeta_upregulated, ageMeta_downregulated, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing AgeMeta signature: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SKIN ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING SKIN DATASET")
        print("="*80)
        
        # Load the skin dataset using YOUR exact code structure
        thresold_senescence = 2
        
        # Load upregulated genes for 40-69 age group
        senescent_human_genes_fibroblast_up_sup40 = pd.read_csv('./Database_age_human_genes/mmc6(1).csv', header=0, delimiter=';')
        senescent_human_genes_fibroblast_up_sup40 = senescent_human_genes_fibroblast_up_sup40.loc[
            (senescent_human_genes_fibroblast_up_sup40['Age'] >= 40) &
            (senescent_human_genes_fibroblast_up_sup40['Age'] <= 69)
        ]
        senescent_human_genes_fibroblast_up_sup40 = senescent_human_genes_fibroblast_up_sup40['gene']
        
        # Load upregulated genes for 70+ age group
        senescent_human_genes_fibroblast_up_sup70 = pd.read_csv('./Database_age_human_genes/mmc6(1).csv', header=0, delimiter=';')
        senescent_human_genes_fibroblast_up_sup70 = senescent_human_genes_fibroblast_up_sup70.loc[senescent_human_genes_fibroblast_up_sup70['Age']>=70]
        senescent_human_genes_fibroblast_up_sup70 = senescent_human_genes_fibroblast_up_sup70['gene']
        
        # Load downregulated genes for 40-69 age group
        thresold_senescence = -2
        senescent_human_genes_fibroblast_down_sup40 = pd.read_csv('./Database_age_human_genes/mmc6_down.csv', header=0, delimiter=';')
        senescent_human_genes_fibroblast_down_sup40 = senescent_human_genes_fibroblast_down_sup40.loc[
            (senescent_human_genes_fibroblast_down_sup40['Age'] >= 40) &
            (senescent_human_genes_fibroblast_down_sup40['Age'] <= 69)
        ]
        senescent_human_genes_fibroblast_down_sup40 = senescent_human_genes_fibroblast_down_sup40['gene']
        
        # Load downregulated genes for 70+ age group
        senescent_human_genes_fibroblast_down_sup70 = pd.read_csv('./Database_age_human_genes/mmc6_down.csv', header=0, delimiter=';')
        senescent_human_genes_fibroblast_down_sup70 = senescent_human_genes_fibroblast_down_sup70.loc[senescent_human_genes_fibroblast_down_sup70['Age']>=70]
        senescent_human_genes_fibroblast_down_sup70 = senescent_human_genes_fibroblast_down_sup70['gene']
        
        # Save the filtered results to separate CSV files in their respective subfolders
        # These will be saved by the analyze_dataset function in the appropriate subfolders
        # No need to save them here as they'll be saved in Skin_40-69/ and Skin_70+/ subfolders
        
        # Display summary of the filtering
        print(f"Number of skin 40-69 upregulated genes: {len(senescent_human_genes_fibroblast_up_sup40)}")
        print(f"Number of skin 70+ upregulated genes: {len(senescent_human_genes_fibroblast_up_sup70)}")
        print(f"Number of skin 40-69 downregulated genes: {len(senescent_human_genes_fibroblast_down_sup40)}")
        print(f"Number of skin 70+ downregulated genes: {len(senescent_human_genes_fibroblast_down_sup70)}")
        print("Skin gene files will be saved in their respective subfolders (Skin_40-69/ and Skin_70+/)")
        
        # Analyze skin 40-69 dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING SKIN 40-69")
        print(f"{'='*60}")
        
        analyze_dataset("Skin_40-69", senescent_human_genes_fibroblast_up_sup40, 
                       senescent_human_genes_fibroblast_down_sup40, age_human_genes, evolutionary_order, results_dir)
        
        # Analyze skin 70+ dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING SKIN 70+")
        print(f"{'='*60}")
        
        analyze_dataset("Skin_70+", senescent_human_genes_fibroblast_up_sup70, 
                       senescent_human_genes_fibroblast_down_sup70, age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing Skin signature: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # OVARY ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING OVARY DATASET")
        print("="*80)
        
        # Load ovary dataset using YOUR exact code structure
        dataset_path = './Database_age_human_genes/ovary_DEGs.csv'
        df = pd.read_csv(dataset_path, sep=';')  # Adjust the delimiter if needed
        
        # Ensure the `avg_log2FC` column exists and convert to numeric
        df['avg_log2FC'] = pd.to_numeric(df['avg_log2FC'].str.replace(',', '.'), errors='coerce')
        
        # Extract upregulated and downregulated genes
        upregulated_genes = df[df['avg_log2FC'] > 0]
        downregulated_genes = df[df['avg_log2FC'] < 0]
        
        # Save the results to CSV files in Ovary subfolder
        ovary_results_dir = os.path.join(results_dir, "Ovary")
        os.makedirs(ovary_results_dir, exist_ok=True)
        
        upregulated_genes.to_csv(os.path.join(ovary_results_dir, "upregulated_genes.csv"), index=False)
        downregulated_genes.to_csv(os.path.join(ovary_results_dir, "downregulated_genes.csv"), index=False)
        
        # Print summary
        print(f"Upregulated genes count: {len(upregulated_genes)}")
        print(f"Downregulated genes count: {len(downregulated_genes)}")
        print(f"Ovary gene files saved to: {ovary_results_dir}")
        
        # Extract gene symbols for analysis
        upregulated_genes = upregulated_genes['gene']
        downregulated_genes = downregulated_genes['gene']
        
        # Analyze ovary dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING OVARY")
        print(f"{'='*60}")
        
        analyze_dataset("Ovary", upregulated_genes, downregulated_genes, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing ovary dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # PROGENITOR CELLS (HPC) ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING PROGENITOR CELLS (HPC) DATASET")
        print("="*80)
        
        # Load progenitor cells dataset using YOUR exact code structure
        dataset_path = './Database_age_human_genes/pone.0005846.s011_HPC.csv'
        df = pd.read_csv(dataset_path, sep=';')  # Adjust the delimiter if needed
        
        # Ensure the `R values` column exists and convert to numeric
        df['R values'] = pd.to_numeric(df['R values'].str.replace(',', '.'), errors='coerce')
        
        # Extract upregulated and downregulated genes
        upregulated_genes = df[df['R values'] > 0]
        downregulated_genes = df[df['R values'] < 0]
        
        # Splitting entries with '///' and exploding into separate rows
        upregulated_genes['Gene Symbol'] = upregulated_genes['Gene Symbol'].str.split(' /// ')
        upregulated_genes = upregulated_genes.explode('Gene Symbol', ignore_index=True)
        downregulated_genes['Gene Symbol'] = downregulated_genes['Gene Symbol'].str.split(' /// ')
        downregulated_genes = downregulated_genes.explode('Gene Symbol', ignore_index=True)
        
        # Save the results to CSV files in Progenitors subfolder
        progenitors_results_dir = os.path.join(results_dir, "Progenitors")
        os.makedirs(progenitors_results_dir, exist_ok=True)
        
        upregulated_genes.to_csv(os.path.join(progenitors_results_dir, "upregulated_genes_HPC.csv"), index=False)
        downregulated_genes.to_csv(os.path.join(progenitors_results_dir, "downregulated_genes_HPC.csv"), index=False)
        
        # Print summary
        print(f"Upregulated genes count: {len(upregulated_genes)}")
        print(f"Downregulated genes count: {len(downregulated_genes)}")
        print(f"Progenitor cells gene files saved to: {progenitors_results_dir}")
        
        # Extract gene symbols for analysis
        upregulated_genes = upregulated_genes['Gene Symbol']
        downregulated_genes = downregulated_genes['Gene Symbol']
        
        # Analyze progenitor cells dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING PROGENITOR CELLS (HPC)")
        print(f"{'='*60}")
        
        analyze_dataset("Progenitors", upregulated_genes, downregulated_genes, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing progenitor cells dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # MESENCHYMAL SENESCENT CELLS ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING MESENCHYMAL SENESCENT CELLS DATASET")
        print("="*80)
        
        # Load mesenchymal senescent cells dataset using YOUR exact code structure
        dataset_path = './Database_age_human_genes/pone.0005846.s010_MSC_senescence.csv'
        df = pd.read_csv(dataset_path, sep=';')  # Adjust the delimiter if needed
        
        # Ensure the `R values` column exists and convert to numeric
        df['R values'] = pd.to_numeric(df['R values'].str.replace(',', '.'), errors='coerce')
        
        # Extract upregulated and downregulated genes
        upregulated_genes = df[df['R values'] > 0]
        downregulated_genes = df[df['R values'] < 0]
        
        # Splitting entries with '///' and exploding into separate rows
        upregulated_genes['Gene Symbol'] = upregulated_genes['Gene Symbol'].str.split(' /// ')
        upregulated_genes = upregulated_genes.explode('Gene Symbol', ignore_index=True)
        downregulated_genes['Gene Symbol'] = downregulated_genes['Gene Symbol'].str.split(' /// ')
        downregulated_genes = downregulated_genes.explode('Gene Symbol', ignore_index=True)
        
        # Save the results to CSV files in Mesenchymal_senescent subfolder
        mesenchymal_senescent_results_dir = os.path.join(results_dir, "Mesenchymal_senescent")
        os.makedirs(mesenchymal_senescent_results_dir, exist_ok=True)
        
        upregulated_genes.to_csv(os.path.join(mesenchymal_senescent_results_dir, "upregulated_genes_MSC_senescence.csv"), index=False)
        downregulated_genes.to_csv(os.path.join(mesenchymal_senescent_results_dir, "downregulated_genes_MSC_senescence.csv"), index=False)
        
        # Print summary
        print(f"Upregulated genes count: {len(upregulated_genes)}")
        print(f"Downregulated genes count: {len(downregulated_genes)}")
        print(f"Mesenchymal senescent cells gene files saved to: {mesenchymal_senescent_results_dir}")
        
        # Extract gene symbols for analysis
        upregulated_genes = upregulated_genes['Gene Symbol']
        downregulated_genes = downregulated_genes['Gene Symbol']
        
        # Analyze mesenchymal senescent cells dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING MESENCHYMAL SENESCENT CELLS")
        print(f"{'='*60}")
        
        analyze_dataset("Mesenchymal_senescent", upregulated_genes, downregulated_genes, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing mesenchymal senescent cells dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # CELLAGE SENESCENCE ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING CELLAGE SENESCENCE DATASET")
        print("="*80)
        
        # Load CellAge senescence dataset using YOUR exact code structure
        dataset_path = './Database_age_human_genes/signatures1.csv'
        df = pd.read_csv(dataset_path, sep=';')  # Adjust the delimiter if needed
        
        # Extract upregulated and downregulated genes
        upregulated_genes = df[df['ovevrexp'] > 0]
        downregulated_genes = df[df['underexp'] > 0]
        
        # Save the results to CSV files in CellAge_Senescence subfolder
        cellage_senescence_results_dir = os.path.join(results_dir, "CellAge_Senescence")
        os.makedirs(cellage_senescence_results_dir, exist_ok=True)
        
        upregulated_genes.to_csv(os.path.join(cellage_senescence_results_dir, "upregulated_genes.csv"), index=False)
        downregulated_genes.to_csv(os.path.join(cellage_senescence_results_dir, "downregulated_genes.csv"), index=False)
        
        # Print summary
        print(f"Upregulated genes count: {len(upregulated_genes)}")
        print(f"Downregulated genes count: {len(downregulated_genes)}")
        print(f"CellAge senescence gene files saved to: {cellage_senescence_results_dir}")
        
        # Extract gene symbols for analysis
        upregulated_genes = upregulated_genes['gene_symbol']
        downregulated_genes = downregulated_genes['gene_symbol']
        
        # Analyze CellAge senescence dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING CELLAGE SENESCENCE")
        print(f"{'='*60}")
        
        analyze_dataset("CellAge_Senescence", upregulated_genes, downregulated_genes, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing CellAge senescence dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # BRAIN ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING BRAIN DATASET")
        print("="*80)
        
        # Create main Brain subfolder
        brain_results_dir = os.path.join(results_dir, "Brain")
        os.makedirs(brain_results_dir, exist_ok=True)
        
        # Data from https://onlinelibrary.wiley.com/doi/10.1111/acel.13280
        
        # ========================================================================
        # BRAIN CORTEX ANALYSIS
        # ========================================================================
        print("\n" + "-"*60)
        print("PROCESSING BRAIN CORTEX")
        print("-"*60)
        
        # Load and clean brain cortex genes using YOUR exact code structure
        old_brain_cortex_human_genes = pd.read_csv('./database_age_human_genes/DEGs_braincortex.csv', header=0, delimiter=';')
        # Replace commas with dots
        old_brain_cortex_human_genes['meanGamma_dts134'] = old_brain_cortex_human_genes['meanGamma_dts134'].str.replace(',', '.')
        # Convert to numeric
        old_brain_cortex_human_genes['meanGamma_dts134'] = pd.to_numeric(old_brain_cortex_human_genes['meanGamma_dts134'], errors='coerce')
        old_brain_cortex_human_genes = old_brain_cortex_human_genes.loc[old_brain_cortex_human_genes['meanGamma_dts134'] >= 0]
        old_brain_cortex_human_genes = old_brain_cortex_human_genes['Gene_Symbol'].dropna()
        
        # Load and clean brain cortex downregulated genes
        down_old_brain_cortex_human_genes = pd.read_csv('./database_age_human_genes/DEGs_braincortex.csv', header=0, delimiter=';')
        # Replace commas with dots
        down_old_brain_cortex_human_genes['meanGamma_dts134'] = down_old_brain_cortex_human_genes['meanGamma_dts134'].str.replace(',', '.')
        # Convert to numeric
        down_old_brain_cortex_human_genes['meanGamma_dts134'] = pd.to_numeric(down_old_brain_cortex_human_genes['meanGamma_dts134'], errors='coerce')
        down_old_brain_cortex_human_genes = down_old_brain_cortex_human_genes.loc[down_old_brain_cortex_human_genes['meanGamma_dts134'] <= 0]
        down_old_brain_cortex_human_genes = down_old_brain_cortex_human_genes['Gene_Symbol'].dropna()
        
        # Create cortex subfolder
        cortex_results_dir = os.path.join(brain_results_dir, "Cortex")
        os.makedirs(cortex_results_dir, exist_ok=True)
        
        # Save cortex genes to CSV
        old_brain_cortex_human_genes.to_csv(os.path.join(cortex_results_dir, "cortex_upregulated_genes.csv"), index=False)
        down_old_brain_cortex_human_genes.to_csv(os.path.join(cortex_results_dir, "cortex_downregulated_genes.csv"), index=False)
        print(f"Brain cortex upregulated genes count: {len(old_brain_cortex_human_genes)}")
        print(f"Brain cortex downregulated genes count: {len(down_old_brain_cortex_human_genes)}")
        print(f"Brain cortex gene files saved to: {cortex_results_dir}")
        
        # Analyze brain cortex dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING BRAIN CORTEX")
        print(f"{'='*60}")
        
        analyze_dataset("Brain_Cortex", old_brain_cortex_human_genes, down_old_brain_cortex_human_genes, 
                       age_human_genes, evolutionary_order, brain_results_dir)
        
        # ========================================================================
        # HIPPOCAMPUS ANALYSIS
        # ========================================================================
        print("\n" + "-"*60)
        print("PROCESSING HIPPOCAMPUS")
        print("-"*60)
        
        # Load and clean hippocampus genes using YOUR exact code structure
        old_hipp_human_genes = pd.read_csv('./database_age_human_genes/DEGs_hipp.csv', header=0, delimiter=';')
        # Replace commas with dots
        old_hipp_human_genes['meanGamma_dts134'] = old_hipp_human_genes['meanGamma_dts134'].str.replace(',', '.')
        # Convert to numeric
        old_hipp_human_genes['meanGamma_dts134'] = pd.to_numeric(old_hipp_human_genes['meanGamma_dts134'], errors='coerce')
        old_hipp_human_genes = old_hipp_human_genes.loc[old_hipp_human_genes['meanGamma_dts134'] >= 0]
        old_hipp_human_genes = old_hipp_human_genes['Gene_Symbol'].dropna()
        
        # Load and clean hippocampus downregulated genes
        down_old_hipp_human_genes = pd.read_csv('./database_age_human_genes/DEGs_hipp.csv', header=0, delimiter=';')
        # Replace commas with dots
        down_old_hipp_human_genes['meanGamma_dts134'] = down_old_hipp_human_genes['meanGamma_dts134'].str.replace(',', '.')
        # Convert to numeric
        down_old_hipp_human_genes['meanGamma_dts134'] = pd.to_numeric(down_old_hipp_human_genes['meanGamma_dts134'], errors='coerce')
        down_old_hipp_human_genes = down_old_hipp_human_genes.loc[down_old_hipp_human_genes['meanGamma_dts134'] <= 0]
        down_old_hipp_human_genes = down_old_hipp_human_genes['Gene_Symbol'].dropna()
        
        # Create hippocampus subfolder
        hipp_results_dir = os.path.join(brain_results_dir, "Hippocampus")
        os.makedirs(hipp_results_dir, exist_ok=True)
        
        # Save hippocampus genes to CSV
        old_hipp_human_genes.to_csv(os.path.join(hipp_results_dir, "hippocampus_upregulated_genes.csv"), index=False)
        down_old_hipp_human_genes.to_csv(os.path.join(hipp_results_dir, "hippocampus_downregulated_genes.csv"), index=False)
        print(f"Hippocampus upregulated genes count: {len(old_hipp_human_genes)}")
        print(f"Hippocampus downregulated genes count: {len(down_old_hipp_human_genes)}")
        print(f"Hippocampus gene files saved to: {hipp_results_dir}")
        
        # Analyze hippocampus dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING HIPPOCAMPUS")
        print(f"{'='*60}")
        
        analyze_dataset("Brain_Hippocampus", old_hipp_human_genes, down_old_hipp_human_genes, 
                       age_human_genes, evolutionary_order, brain_results_dir)
        
        # ========================================================================
        # CEREBELLUM ANALYSIS
        # ========================================================================
        print("\n" + "-"*60)
        print("PROCESSING CEREBELLUM")
        print("-"*60)
        
        # Load and clean cerebellum genes using YOUR exact code structure
        old_brain_cerebell_genes = pd.read_csv('./database_age_human_genes/DEGs_cerebellum.csv', header=0, delimiter=';')
        # Replace commas with dots
        old_brain_cerebell_genes['meanGamma_dts13'] = old_brain_cerebell_genes['meanGamma_dts13'].str.replace(',', '.')
        # Convert to numeric
        old_brain_cerebell_genes['meanGamma_dts13'] = pd.to_numeric(old_brain_cerebell_genes['meanGamma_dts13'], errors='coerce')
        old_brain_cerebell_genes = old_brain_cerebell_genes.loc[old_brain_cerebell_genes['meanGamma_dts13'] >= 0]
        old_brain_cerebell_genes = old_brain_cerebell_genes['Gene_Symbol'].dropna()
        
        # Load and clean cerebellum downregulated genes
        down_old_brain_cerebell_genes = pd.read_csv('./database_age_human_genes/DEGs_cerebellum.csv', header=0, delimiter=';')
        # Replace commas with dots
        down_old_brain_cerebell_genes['meanGamma_dts13'] = down_old_brain_cerebell_genes['meanGamma_dts13'].str.replace(',', '.')
        # Convert to numeric
        down_old_brain_cerebell_genes['meanGamma_dts13'] = pd.to_numeric(down_old_brain_cerebell_genes['meanGamma_dts13'], errors='coerce')
        down_old_brain_cerebell_genes = down_old_brain_cerebell_genes.loc[down_old_brain_cerebell_genes['meanGamma_dts13'] <= 0]
        down_old_brain_cerebell_genes = down_old_brain_cerebell_genes['Gene_Symbol'].dropna()
        
        # Create cerebellum subfolder
        cerebell_results_dir = os.path.join(brain_results_dir, "Cerebellum")
        os.makedirs(cerebell_results_dir, exist_ok=True)
        
        # Save cerebellum genes to CSV
        old_brain_cerebell_genes.to_csv(os.path.join(cerebell_results_dir, "cerebellum_upregulated_genes.csv"), index=False)
        down_old_brain_cerebell_genes.to_csv(os.path.join(cerebell_results_dir, "cerebellum_downregulated_genes.csv"), index=False)
        print(f"Cerebellum upregulated genes count: {len(old_brain_cerebell_genes)}")
        print(f"Cerebellum downregulated genes count: {len(down_old_brain_cerebell_genes)}")
        print(f"Cerebellum gene files saved to: {cerebell_results_dir}")
        
        # Analyze cerebellum dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING CEREBELLUM")
        print(f"{'='*60}")
        
        analyze_dataset("Brain_Cerebellum", old_brain_cerebell_genes, down_old_brain_cerebell_genes, 
                       age_human_genes, evolutionary_order, brain_results_dir)
        
    except Exception as e:
        print(f"Error analyzing brain dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # MESENCHYMAL STEM CELLS ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING MESENCHYMAL STEM CELLS DATASET")
        print("="*80)
        
        # Load mesenchymal stem cells dataset using YOUR exact code structure
        dataset_path = './Database_age_human_genes/pone.0005846.s009.csv'
        df = pd.read_csv(dataset_path, sep=';')  # Adjust the delimiter if needed
        
        # Ensure the `R values` column exists and convert to numeric
        df['R values'] = pd.to_numeric(df['R values'].str.replace(',', '.'), errors='coerce')
        
        # Extract upregulated and downregulated genes
        upregulated_genes = df[df['R values'] > 0]
        downregulated_genes = df[df['R values'] < 0]
        
        # Splitting entries with '///' and exploding into separate rows
        upregulated_genes['Gene Symbol'] = upregulated_genes['Gene Symbol'].str.split(' /// ')
        upregulated_genes = upregulated_genes.explode('Gene Symbol', ignore_index=True)
        downregulated_genes['Gene Symbol'] = downregulated_genes['Gene Symbol'].str.split(' /// ')
        downregulated_genes = downregulated_genes.explode('Gene Symbol', ignore_index=True)
        
        # Create mesenchymal stem cells subfolder
        mesenchymal_results_dir = os.path.join(results_dir, "Mesenchymal")
        os.makedirs(mesenchymal_results_dir, exist_ok=True)
        
        # Save the results to CSV files
        upregulated_genes.to_csv(os.path.join(mesenchymal_results_dir, "upregulated_genes_MSC.csv"), index=False)
        downregulated_genes.to_csv(os.path.join(mesenchymal_results_dir, "downregulated_genes_MSC.csv"), index=False)
        
        # Print summary
        print(f"Upregulated genes count: {len(upregulated_genes)}")
        print(f"Downregulated genes count: {len(downregulated_genes)}")
        print(f"Mesenchymal stem cells gene files saved to: {mesenchymal_results_dir}")
        
        # Extract gene symbols for analysis
        upregulated_genes = upregulated_genes['Gene Symbol']
        downregulated_genes = downregulated_genes['Gene Symbol']
        
        # Analyze mesenchymal stem cells dataset (upregulated + downregulated)
        print(f"\n{'='*60}")
        print(f"ANALYZING MESENCHYMAL STEM CELLS")
        print(f"{'='*60}")
        
        analyze_dataset("Mesenchymal", upregulated_genes, downregulated_genes, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing mesenchymal stem cells dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # CD8+ T CELLS ANALYSIS
    # ========================================================================
    try:
        print("\n" + "="*80)
        print("PROCESSING CD8+ T CELLS DATASET")
        print("="*80)
        
        # Load CD8+ T cells data using YOUR exact code structure
        cd8t_upregulated = pd.read_csv('./Database_age_human_genes/over_CD8T.csv', header=0, delimiter=';')
        cd8t_upregulated = cd8t_upregulated['Gene symbol'].dropna()
        
        cd8t_downregulated = pd.read_csv('./Database_age_human_genes/down_CD8T.csv', header=0, delimiter=';')
        cd8t_downregulated = cd8t_downregulated['Gene symbol'].dropna()
        
        print(f"Loaded CD8+ T cells signature: {len(cd8t_upregulated)} upregulated, {len(cd8t_downregulated)} downregulated")
        
        # Analyze CD8+ T cells dataset
        analyze_dataset("CD8T", cd8t_upregulated, cd8t_downregulated, 
                       age_human_genes, evolutionary_order, results_dir)
        
    except Exception as e:
        print(f"Error analyzing CD8+ T cells: {e}")
        import traceback
        traceback.print_exc()
    
    # Create Mann-Whitney U test results table
    print("\n" + "="*80)
    print("CREATING MANN-WHITNEY U TEST RESULTS TABLE")
    print("="*80)
    create_mann_whitney_results_table(results_dir)
    
    # Create combined results table figure
    print("\n" + "="*80)
    print("CREATING COMBINED RESULTS TABLE FIGURE")
    print("="*80)
    create_combined_results_table_figure(results_dir)
    
    print(f"\nEnhanced multi-dataset analysis complete! Results saved in {results_dir}/")
    print("Analysis includes:")
    print("1. CellAge dataset (upregulated, downregulated, combined)")
    print("2. AgeMeta dataset (upregulated, downregulated, combined)")
    print("3. Skin dataset (upregulated, downregulated, combined)")
    print("4. Ovary dataset (upregulated, downregulated, combined)")
    print("5. Progenitor cells dataset (upregulated, downregulated, combined)")
    print("6. Mesenchymal senescent cells dataset (upregulated, downregulated, combined)")
    print("7. CellAge senescence dataset (upregulated, downregulated, combined)")
    print("8. Brain dataset (cortex, hippocampus, cerebellum)")
    print("9. Mesenchymal stem cells dataset (upregulated, downregulated, combined)")
    print("10. CD8+ T cells dataset (upregulated, downregulated, combined)")
    print("11. Raw counts and percentages")
    print("12. Over and under-representation tests")
    print("13. Enhanced stacked histograms")
    print("14. Results organized in separate subfolders")
    print("15. Mann-Whitney U test results table (Mann_Whitney_Results_Table.csv)")

if __name__ == "__main__":
    main() 