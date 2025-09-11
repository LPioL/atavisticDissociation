# GenAge Atavism Analysis

This repository contains the analysis code for studying evolutionary age patterns in GenAge (CellAge) gene sets using atavism theory principles.

## Overview

The analysis examines whether aging-related gene sets show patterns consistent with atavism theory - the idea that aging involves the re-emergence of ancient evolutionary traits. We analyze gene sets for over/under-representation across evolutionary age categories and perform statistical tests to assess significance.

## Files

### Main Analysis Script
- `genage_percentages_only.py` - Main analysis script that creates stacked percentage plots (second subfigure from original analysis) for GenAge data

### Data Files
- `data/` - Contains the required data files:
  - `over.all.csv` - GenAge upregulated genes
  - `under.all.csv` - GenAge downregulated genes  
  - `1-s2.0-S0093775418302264-mmc2.csv` - Baseline genome evolutionary age data

### Results
- `results/` - Contains analysis outputs:
  - `genage_upregulated_percentages.png` - Upregulated genes analysis
  - `genage_downregulated_percentages.png` - Downregulated genes analysis
  - `genage_combined_percentages.png` - Combined genes analysis
  - `genage_mann_whitney_results.csv` - Mann-Whitney U test results

## Requirements

```bash
pip install pandas numpy matplotlib scipy statsmodels
```

## Usage

1. Place the required data files in the `data/` directory
2. Run the analysis:
```bash
python genage_percentages_only.py
```

## Analysis Details

### Statistical Tests
- **Hypergeometric Test**: Tests for over/under-representation of each evolutionary age category
- **FDR Correction**: Benjamini-Hochberg correction for multiple testing
- **Mann-Whitney U Test**: Compares overall age distribution between gene set and baseline

### Evolutionary Age Categories
The analysis uses 19 evolutionary age categories (ranks 1-19):
1. All living organisms (oldest)
2. Eukaryota
3. Opisthokonta
4. Holozoa
5. Metazoa
6. Eumetazoa
7. Bilateria
8. Deuterostomia
9. Chordata
10. Olfactores
11. Craniata
12. Euteleostomi
13. Tetrapoda
14. Amniota
15. Mammalia
16. Eutheria
17. Boreoeutheria
18. Euarchontoglires
19. Primates (youngest)

### Output Interpretation
- **Red stars**: Over-represented categories (FDR < 0.05, fold change > 1.1)
- **Black stars**: Under-represented categories (FDR < 0.05, fold change < 0.9)
- **Mean age shift**: Negative values indicate shift toward older ages

## Citation

If you use this code, please cite the original paper and acknowledge the atavism theory framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
