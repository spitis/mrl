These are some files for plotting tensorboard results.

## The steps are:
1. Use `exportTB.py` to recursively extract from any tensorboard event files the scalar value CSVs in the output folder. This writes all the CSVs in a flat folder with the relative folder path concatenated with the summary tag. **UPDATE: The logger now saves CSVs that you can use directly; no extraction from TB necessary.**
   * Command Template: `python exportTB.py <output-folder> <output-path-to-csv> <summaries>`
   * Example: `python exportTB.py /path/to/folder/with/tensorboard /path/to/export/folder Test/Success,Explore/ag_kde_entropy,Explore/Intrinsic_success_percent,Explore/curiosity_alpha`
2. Manually separate out the CSV files into a `sources` folder. For example, a folder represents one experiment and can contain several CSV files for different seeds, each plotting accuracy vs training step. This is the following hierarchy:
```
   sources
   -- plot_1
        -- curve_1
            -- seed1.csv
            -- seed2.csv
        -- curve_2
            -- seed1.csv
            -- seed2.csv
```
3. If you have some updated tensorboard event files and need to reexport, you can run #1 step above again. But instead of manually copying over, just run `gather_sources.py /path/to/exported` in order to update the newly exported CSV file into the folder hierarchy above.  
