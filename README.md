# Data Backup Bundle

This repository keeps the runnable script bundle for the main final-target experiment. The cleaned final data archive is distributed separately and is not tracked in Git.

## Contents
- `scripts/`: minimal runnable code bundle for the main experiment.

## Data Download
The cleaned final experiment archive and extracted runtime data are available via OneDrive:

- archive download: [OneDrive link](https://1drv.ms/u/c/5e124d2f43c55892/IQCqk_FU5jUkSbLsGlZntRLLATiT76ZKJhNSlj3Y9DAQUvQ?e=chHuEr)
- files provided there:
  - `final_target_20260311.tar.gz`
  - `final_target_20260311.tar.gz.sha256`
- this repository does not track the archive itself or the extracted `scripts/experiment_videos/` data directory

## Scripts Bundle
The `scripts/` folder includes the files used by the final main experiment:
- `run_main_final_target_experiment.sh`: public-facing wrapper for the final run.
- `run_experiment.py`: experiment orchestrator.
- `model_config.py`: target-model-only configuration used by the released bundle.
- `prompt_factory.py`, `prompt_templates.py`, `answer_parser.py`
- `questions.json`, `detailprompt.csv`
- the 9 model chat entry scripts used by the final run.

## Local Data Layout
If you unpack the released data bundle locally, place the extracted experiment videos under `scripts/experiment_videos/` before rerunning the experiment.
