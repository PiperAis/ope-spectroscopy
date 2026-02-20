README
Pull this repo directly into the Code folder inside your project.

Full_TRR_Report_Generator.py is the only script you need to interact with in order to process transient reflectance data. 

config.py sets up relative paths between code repo, data, and markdown data processing reports. The expected file strucuture is
```
.
├── Code/  
│   ├── processing_package  
│   ├── config.py  
│   └── Full_TRR_Report_Generator.py  
└── TRR/  
    └── raw
    │   ├── 2026-02-15
    └   └── 2026-02-16
```
Where dated folders contain raw data collected using TimeSpec software.

Executing Full_TRR_Report_generator.py will generate other subfolders and store modified data and the markdown report in them.

