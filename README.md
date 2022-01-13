**Epigame package available at: https://github.com/ivkarla/epigame**

# epigame-y1
A project for epileptogenic network definition: implementation of a seizure generation model based on a game theory principle and connectivity dynamics.

## _DATA_:

SEEG data should include the signal of 10 minutes preceding the clinical annotation of seizure end. Seizure start annotation should be named _“EEG inicio”_, and seizure end _“EEG fin”_. For baseline analysis (interictal state far from ictal state), data should include the signal of 10 minutes. Annotation should be placed at the recording end (_“EEG fin”_) and middle (_“EEG inicio”_). SEEG signal and metadata should be extracted as EDF+ file. The filenames must start with a three-letter patient identifier, e.g., "ABC".

## _FILE SYSTEM_:

Create the main directory with two subdirectories: _code_ and _data_. Place the main pipeline and all the dependencies in _code_. Create three directories within data: _raw, preprocessed_ and _results_. Place the EDF raw data files in _raw_. _Preprocessed_ and _results_ will contain the preprocessed data and analysis results, respectively.
Create directories for connectivity analysis type in _preprocessed_ and _results_: _CC, PAC, PEC, PLI, PLV, SC_R, SC_I_. Within _CC, PLI, PLV, SC_R_ and _SC_I_, create directories for specific frequency bands: _(1,4), (4,8), (8,12), (13,30), (30,70), (70,150)_.
If you wish to compute multiscores, all of the aforementioned directories are necessary. If you are interested in a specific connectivity analysis, simply create only the corresponding directory.
