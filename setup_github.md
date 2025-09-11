# GitHub Setup Instructions

## 1. Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: GenAge Atavism Analysis"

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/genage-atavism-analysis.git

# Push to GitHub
git push -u origin main
```

## 2. Repository Structure

The repository is organized as follows:

```
genage-atavism-analysis/
├── README.md                           # Main documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
├── genage_percentages_only.py          # Main analysis script
├── data/                               # Data files
│   ├── over.all.csv                   # GenAge upregulated genes
│   ├── under.all.csv                  # GenAge downregulated genes
│   └── 1-s2.0-S0093775418302264-mmc2.csv  # Baseline genome data
└── results/                            # Analysis outputs
    ├── genage_upregulated_percentages.png
    ├── genage_downregulated_percentages.png
    ├── genage_combined_percentages.png
    └── genage_mann_whitney_results.csv
```

## 3. Usage Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genage-atavism-analysis.git
cd genage-atavism-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python genage_percentages_only.py
```

## 4. Key Features

- **Clean, focused code** - Only the essential analysis
- **Proper data organization** - Data files in dedicated folder
- **Complete documentation** - README with usage instructions
- **Reproducible results** - All dependencies specified
- **Professional structure** - Follows GitHub best practices

## 5. Files Included

- **Main script**: `genage_percentages_only.py` - Creates stacked percentage plots
- **Data files**: All required data files in `data/` folder
- **Results**: Sample outputs in `results/` folder
- **Documentation**: Complete README and setup instructions
- **Dependencies**: `requirements.txt` with all needed packages
