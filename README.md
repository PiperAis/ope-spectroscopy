README


Full_TRR_Report_Generator.py is the only script you need to interact with in order to process transient reflectance data. 

config.py sets up relative paths between code repo, data, and markdown data processing reports. The expected file strucuture is
```
Project root
├── config.py 
├── PL/ 
    └   └── 2026-01-28
└── TRR/  
    └── raw
    │   ├── 2026-02-15
    └   └── 2026-02-16
```
Where dated folders contain raw data (for TRR, collected using TimeSpec software).

Executing Full_TRR_Report_generator.py will generate other subfolders and store modified data and the markdown report in them.

