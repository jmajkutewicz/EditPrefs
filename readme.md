# Wiki Preferences

This repo contains code for paper "TODO"

This tools creates LLM alignment dataset from Wikipedia articles' edition history.

## Dataset

The Wiki Preferences dataset created from English Wikipedia dump from 01.04.2024 is available on HuggingFace: https://huggingface.co/datasets/jmajkutewicz/WikiPrefs

## Usage

Note that the repository uses `git lfs` to store resource files. Make sure you have `git lfs` installed.

### Setup

Creating Conda environment and installing required dependencies:

```bash
conda env create -f wiki_prefs.yml
conda activate wiki_prefs

python -m spacy download en_core_web_sm
```

### Creating dataset from Featured Articles edits

1. Download featured articles list from Wikipedia:
    ```bash
   python -m wikiprefs.download_featured_articles_list
    ```
2. Extract all featured articles edits from Wikipedia meta-history dump:
    ```bash
   python -m wikiprefs.download_articles_history \
      --dst path_to_directory_for_saving_featured_articles_history
    ```
   Note that we don't save whole meta-history dump to disk, but only featured articles history. This allows to minimize
   disk space requirements.
3. Extract diffs from featured articles history to csv files
    ```bash
   python -m wikiprefs.collect_retained_diffs \
      --src path_to_directory_with_saved_featured_articles_history\
      --dst path_to_directory_for_saving_extracted_diffs
    ```
4. Create dataset from the extracted diffs
    ```bash
   python -m wikiprefs.create_dataset \
      -t fa \
      --src path_to_directory_with_extracted_diffs \
      --hf-repo huggingface_repo_id
    ```

### Creating dataset from NPOV edits

1. Find revisions, which comments suggest NPOV edits:
    ```bash
   python -m wikiprefs.find_npov_edits \
      --stub-files-dir path_to_directory_for_saving_history_stub_archives
    ```
2. Process Wikipedia dump and extract diffs for the NPOV edits:
    ```bash
   python -m wikiprefs.collect_npov_diffs \
    --dst path_to_directory_for_saving_extracted_diffs
    ```
3. Create dataset from the extracted diffs
    ```bash
   python -m wikiprefs.create_dataset \
      -t npov \
      --src path_to_directory_with_extracted_npov_diffs \
      --hf-repo huggingface_repo_id
    ```

## Making changes

To validate your changes:

* run linter: `./run_lint.sh`
* then run tests: `./run_tests.sh`

### Linter & formatter

We're using Ruff: <https://github.com/astral-sh/ruff>  
Docstrings format: <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>

## Citation

Please cite our paper if you use in this software or dataset in your research.

```
TODO
```