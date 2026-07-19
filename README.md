# A solution to generalized learning from small training sets found in infants’ repeated visual experiences of individual objects

Thanks for your interest in our work! Please feel free to reach out via email if you need anything.
The infant head-camera images have been deposited to Databrary at: (TBA)

## Repository structure

```text
.
├── Fig1/            # Object-ID rank frequency plot
│   ├── code/        
│   └── data/  
|      
├── Fig2/            # Similarity rainplots and graph networks
│   ├── code/
│   │   ├── PanelA/  # Image feature extraction, null distribution generation, infant vs. null similarity rainplots, and infant rainplots by group (R1, etc.)
│   │   ├── PanelB/  # Graph metrics
│   │   └── PanelC/  # Plotting graph results
│   └── data/
│       ├── PanelA/ 
│       └── PanelC/  
|
├── Fig3/            # Machine learning experiments
│   ├── code/        # Model training and result figures
│   └── data/ 
|       
├── Fig4/            # Intentionally empty
|
├── FigS1/           # More frequency statistics, similar to Fig1
│   ├── PanelA/      # Objects per frame
│   ├── PanelB/      # Object-ID rank frequency plot within a mealtime
│   └── PanelC/      # Intentionally empty
|
├── FigS3/           # Similarity rainplots by category, by infant, and entropy plots
│   ├── PanelsAC/    # Intentionally empty
│   └── PanelB/      # Intentionally empty
|
├── FigS2/           # Graph ablations by visual feature and by edge similarity threshold 
│   └── PanelsAB/    # Graph metrics by feature (RGB histogram, GIST, PE-Spatial, CLIP)
│   └── PanelCD/     # Graph metrics by edge threshold (top 5, 10, and 20% similarities)
|
├── FigS4/           # Blocks of Shuffled Labels graph null block-size ablation
|
├── FigS5-to-S8/     # Per-subject graph connectivity measures
|
└── TableS1/         # Intentionally empty
|
└── TablesS2S3/      # Machine learning lumpiness analysis
|   ├── code/        # Image feature extraction, graph metrics, plots
|   └── data/   
|
└── TableS4/         # Intentionally empty
```

## Notes

- The codebase should be relatively self-explanatory if one follows the paper figures. Please note some scripts require the user to change the input/output CSV filenames manually. For example, in `Fig2/code/PanelB/`, one can change from RGB histogram to CLIP features by appending the suffix `_withClip` to the CSV input filenames (which are available in their corresponding `data/` folder for guidance).
- The released images have a filename suffix of either `_bbox.jpg` or `_manual.jpg`. This is to indicate we masked visible faces to protect the privacy of human participants, as per our IRB regulations (see our paper for more details). The suffix is ignored by our scripts.