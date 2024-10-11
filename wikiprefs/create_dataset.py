import argparse
import asyncio
import csv
import logging
import os
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from wikiprefs.diffs.filtering import (
    BleuFilter,
    CommentFilter,
    LengthFilter,
    OutliersFilter,
    RemovalFilter,
    SingleEditFilter,
    WikiMarkupFilter,
)
from wikiprefs.markup.caching_template_renderer import CachingTemplateRenderer
from wikiprefs.utils.csv_utils import fix_field_size_limit
from wikiprefs.utils.log_utils import setup_logger

logger = setup_logger(log_level=logging.INFO, filename='create_dataset.log')


def get_diff_files(src_dir: str) -> [str]:
    """Get all source CSV files containing diffs for each article"""
    diff_files = []
    for f in os.listdir(src_dir):
        if not os.path.isfile(os.path.join(src_dir, f)):
            continue
        if not f.endswith('.csv'):
            continue
        diff_files.append(f)

    return diff_files


def merge_diffs(src_dir, diff_files: [str], destination_csv_file_path: Path) -> None:
    """Merge all diffs into a single CSV file"""
    logger.info(f'Merging {len(diff_files)} diffs to {destination_csv_file_path}')
    length_filter = LengthFilter()
    diff_total_count = 0
    diffs_saved = 0

    with open(os.path.join(src_dir, diff_files[0]), newline='', encoding='utf-8') as csvfile:
        fieldnames = csv.DictReader(csvfile).fieldnames
    logger.info(f'Fieldnames: {fieldnames}')

    with open(destination_csv_file_path, mode='w', newline='', encoding='utf-8') as destination_csv:
        writer = csv.DictWriter(destination_csv, fieldnames=fieldnames)
        writer.writeheader()

        for i, diff_file in enumerate(diff_files):
            unique_old_text = set()
            with open(os.path.join(src_dir, diff_file), newline='', encoding='utf-8') as source_file:
                reader = csv.DictReader(source_file)
                row: dict[str, Any]
                for row in reader:
                    old_text = row['old_text']
                    new_text = row['new_text']

                    if old_text in unique_old_text:
                        continue
                    unique_old_text.add(old_text)

                    if not length_filter.is_too_long(old_text, new_text):
                        writer.writerow(row)
                        diffs_saved += 1
                    diff_total_count += 1

            if i % 10 == 0:
                logger.info(f'Merged {i} files')

    logger.info(f'Saved {diffs_saved} out of {diff_total_count}')


def get_outliers_filter(config) -> OutliersFilter:
    """Get Outliers Filter configured for the current dataset type"""
    if config.dataset_type == 'npov':
        # all comments already suggest NPOV edit, so we can ignore them
        outliers_filter = OutliersFilter(
            comment_filter=lambda _: False, max_relative_length_difference_quantile=0.95, filter_out_simple_edits=True
        )
    elif config.dataset_type == 'fa':
        # less aggressive filtering out for FA edits since we're interested in all text improvements
        comments_filter = CommentFilter()
        outliers_filter = OutliersFilter(
            comment_filter=lambda c: comments_filter.is_fa_edit(c),
            max_relative_length_difference_quantile=0.99,
            filter_out_simple_edits=False,
        )
    else:
        logger.error(f'Invalid dataset type {config.dataset_type}')
        exit(1)

    return outliers_filter


def bleu_threshold__hardcoded(_: pd.Series) -> float:
    """Returns hardcoded bleu threshold"""
    return 0.15


def bleu_threshold__first_quantile(bleu_score: pd.Series) -> float:
    """Returns bleu 1st quantile as the threshold"""
    return bleu_score.quantile(0.01)


def get_bleu_filter(config) -> BleuFilter:
    """Get Bleu Filter configured for the current dataset type"""
    if config.dataset_type == 'npov':
        # hardcoded bleu score for npov diffs to improve quality of dataset
        bleu_filter = BleuFilter(bleu_threshold__hardcoded)
    elif config.dataset_type == 'fa':
        # less aggressive filtering out for FA edits
        bleu_filter = BleuFilter(bleu_threshold__first_quantile)
    else:
        logger.error(f'Invalid dataset type {config.dataset_type}')
        exit(1)

    return bleu_filter


def create_huggingface_dataset(
    data: pd.DataFrame, tmp_dir: Path, dst_dir: str | None, hugging_face_repo_id: str | None, test_size: float | int
) -> None:
    """Save the diffs as a Huggingface dataset

    Args:
        data: pandas dataframe with the dataset data
        tmp_dir: path to temporary directory for storing intermediate files
        dst_dir: optional destination directory to save the Huggingface dataset
        hugging_face_repo_id: optional Huggingface repository id for publishing the dataset on the Hub
        test_size: size of the test split of the dataset
    """
    dpo_data = pd.DataFrame(
        {
            'page_id': data['page_id'],
            'page_title': data['page_title'],
            'section': data['section'],
            'rev_id': data['rev_id'],
            'prev_rev_id': data['prev_rev_id'],
            'timestamp': data['timestamp'],
            'contributor': data['contributor'],
            'comment': data['comment'],
            'prompt': data['prompt'],
            'chosen': data.apply(
                lambda x: [{'content': x['prompt'], 'role': 'user'}, {'content': x['new_text'], 'role': 'assistant'}],
                axis=1,
            ),
            'rejected': data.apply(
                lambda x: [{'content': x['prompt'], 'role': 'user'}, {'content': x['old_text'], 'role': 'assistant'}],
                axis=1,
            ),
        }
    )

    train, test = train_test_split(dpo_data, test_size=test_size, random_state=42)

    train.to_json(str(tmp_dir / 'train_wikiprefs.json'), orient='records', lines=True)
    test.to_json(str(tmp_dir / 'test_wikiprefs.json'), orient='records', lines=True)
    dataset = load_dataset(str(tmp_dir), data_files={'train': 'train_wikiprefs.json', 'test': 'test_wikiprefs.json'})

    if hugging_face_repo_id:
        logger.info(f'Publishing dataset on HuggingFace hub as {hugging_face_repo_id}')
        dataset.push_to_hub(hugging_face_repo_id)
    if dst_dir:
        logger.info(f'Saving HuggingFace dataset to {dst_dir}')
        dataset.save_to_disk(os.path.join(dst_dir, 'huggingface_dataset'))


async def render_templates_for_row_async(
    template_renderer: CachingTemplateRenderer,
    semaphore: asyncio.Semaphore,
    df: pd.DataFrame,
    index: Hashable,
    row: pd.Series,
) -> Hashable | None:
    """Render Wikipedia markup templates in the text of given row"""
    async with semaphore:
        logger.debug(f'Start processing row {index}')
        old_text = row['old_text']
        new_text = row['new_text']

        if old_text:
            old_text_updated, old_text = await template_renderer.process_templates(old_text)
            if old_text is None:
                logger.warning(f"Dropping row {index}, because old text can't be rendered")
                return index
            if old_text_updated:
                df.at[index, 'old_text'] = old_text

        if new_text:
            new_text_updated, new_text = await template_renderer.process_templates(new_text)
            if new_text is None:
                logger.warning(f"Dropping row {index}, because new text can't be rendered")
                return index
            if new_text_updated:
                df.at[index, 'new_text'] = new_text

        if (old_text_updated or new_text_updated) and (new_text == old_text):
            logger.warning(f'After rendering templates, old and new text is the same (row {index})')
            return index

        logger.debug(f'Completed processing row {index}')
        return None


async def render_templates_async(df: pd.DataFrame):
    """Render all Wikipedia markup templates"""
    df.reset_index(inplace=True, drop=True)  # make sure indexes pair with number of rows
    semaphore = asyncio.Semaphore(10)
    async with CachingTemplateRenderer() as template_renderer:
        tasks = [
            render_templates_for_row_async(template_renderer, semaphore, df, index, row) for index, row in df.iterrows()
        ]
        invalid_rows = await asyncio.gather(*tasks)
        invalid_rows = [i for i in invalid_rows if i is not None]

        logger.info(f'Send {template_renderer.requests_send} requests to Wikipedia API')

    logger.warning(f'Dropping {len(invalid_rows)} invalid rows out of {df.shape[0]} rows')
    df.drop(index=invalid_rows, inplace=True)
    logger.info(f'Remaining rows: {df.shape[0]}')


def render_templates(df):
    """Render all Wikipedia markup templates asynchronously"""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(render_templates_async(df))
    finally:
        loop.close()


class PromptGenerationException(Exception):
    """Error thrown when prompt generation fails"""

    def __init__(self, message: str):
        """Initialize exception"""
        super().__init__(message)


def generate_prompts(df: pandas.DataFrame, prompt_gen: str, get_tmp_file: Callable[[str], Path]) -> pandas.DataFrame:
    """Generate prompts(instruction/questions) for the text"""
    # import inside function for injecting mock class in tests:
    from wikiprefs.prompts.gpt3 import Gpt3PromptGenerator

    prompt_column = 'prompt'
    tmp_file = get_tmp_file('prompt_tmp')
    if os.path.exists(tmp_file):
        df = pd.read_csv(tmp_file)
        df = df.fillna('')
    else:
        df[prompt_column] = ''
        df.to_csv(tmp_file, index=False)

    if prompt_gen == 'gpt3':
        logger.info('Using GPT-3.5 for prompt generation')
        generator = Gpt3PromptGenerator()

        def generate_prompt(row):
            try:
                return generator.generate_prompt(row['page_title'], row['section'], row['new_text'])
            except Exception as e:
                logger.error(f'Exception while generating prompt for {row["page_title"]} - {row["section"]}: {e}')
                return ''

        rows_to_process = df[df[prompt_column] == '']
        logger.info(f'There are {rows_to_process.shape[0]} rows to process')
        # rows_to_process = rows_to_process.sample(min(len(rows_to_process), 1000))  # this is for tests only

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_row = {executor.submit(generate_prompt, row): index for index, row in rows_to_process.iterrows()}

            count = 0
            for future in as_completed(future_to_row):
                index = future_to_row[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f'Exception while fetching prompt from future: {e}')
                    continue

                df.at[index, prompt_column] = result
                # Save the DataFrame every 1000 completed prompts
                count += 1
                if count % 1000 == 0:
                    df.to_csv(tmp_file, index=False)
                    logger.info(f'Saved progress after {count} prompts')

        logger.info('Finished generating prompts via gpt3')
    elif prompt_gen == 'simple':
        logger.info('Using simple prompt generation strategy')
        df[prompt_column] = 'Write something about ' + df['page_title'] + ' - ' + df['section']
    else:
        logger.error(f'Invalid prompt generation strategy {prompt_gen}')
        raise ValueError(f'Invalid prompt generation strategy {prompt_gen}')

    df.to_csv(tmp_file, index=False)

    empty_prompt_rows = df[df['prompt'] == '']
    empty_prompt_rows_count = empty_prompt_rows.shape[0]
    if empty_prompt_rows_count > 0:
        logger.error(
            f'Failed to generate prompt for following {empty_prompt_rows_count} rows. '
            f'Resolve the underlying issue and run the generation again'
        )
        logger.error(f'{empty_prompt_rows.index.tolist()}')
        raise PromptGenerationException(f'Failed to generate prompts for {empty_prompt_rows_count} rows')

    return df


def main():
    """Main"""
    parser = setup_argument_parser()
    config = parser.parse_args()
    print(f'Config: {config}')
    if not os.path.isdir(config.src):
        logger.error(f'Source directory {config.src} does not exist')
        exit(1)
    if not config.dst and not config.hf_repo:
        logger.error('Destination directory or HuggingFace repo is required')
        exit(1)
    get_tmp_file = get_tmp_file_generator(config.dataset_type)

    ds_merged_file = get_tmp_file('ds_merged')
    if not os.path.exists(ds_merged_file):
        diff_files = get_diff_files(config.src)
        logger.info(f'Merging {len(diff_files)} diffs to {ds_merged_file}')
        merge_diffs(config.src, diff_files, ds_merged_file)

    ds_filtered_file = get_tmp_file('ds_filtered')
    if not os.path.exists(ds_filtered_file):
        data = pd.read_csv(ds_merged_file, na_values='')
        data.fillna('', inplace=True)

        if config.dataset_type == 'npov':
            # filter out revisions that edited more than 1 paragraph only for npov diffs
            # to ensure we have edits that are improving NPOV and not unrelated edits in the same revision
            logger.info(f'Filtering out revisions with multiple edits (rows: {data.shape[0]})')
            single_edit_filter = SingleEditFilter()
            data = single_edit_filter.filter_diffs(data)
            logger.info(f'Finished filtering out revisions with multiple edits (rows: {data.shape[0]})')

        logger.info(f'Filtering out outliers (rows: {data.shape[0]})')
        # remove outliers before rendering templates, to limit the number text to render
        # thus limit number of requests we need to send to Wikipedia API
        outliers_filter = get_outliers_filter(config)
        data = outliers_filter.filter_diffs(data)
        logger.info(f'Finished filtering out outliers (rows: {data.shape[0]})')

        data.to_csv(ds_filtered_file, index=False)

    ds_rendered_file = get_tmp_file('ds_rendered')
    if not os.path.exists(ds_rendered_file):
        data = pd.read_csv(ds_filtered_file, na_values='')
        data.fillna('', inplace=True)

        logger.info(f'Rendering templates (rows: {data.shape[0]})')
        render_templates(data)
        logger.info(f'Finished rendering (rows: {data.shape[0]})')

        data.to_csv(ds_rendered_file, index=False)

    ds_cleaned_file = get_tmp_file('ds_cleaned')
    if not os.path.exists(ds_cleaned_file):
        data = pd.read_csv(ds_rendered_file, na_values='')
        data.fillna('', inplace=True)

        logger.info(f'Filtering out rows with remaining wiki markup (rows: {data.shape[0]})')
        markup_filter = WikiMarkupFilter()
        data = markup_filter.filter_diffs(data)
        logger.info(f'Finished filtering out rows with remaining wiki markup (rows: {data.shape[0]})')

        if config.dataset_type == 'npov':
            pass

        logger.info(f'Filtering out outliers after template rendering (rows: {data.shape[0]})')
        # repeat the outliers filtering, as the text have changed after templates rendering,
        # so there are no guarantees that the text still meets all criteria
        outliers_filter = get_outliers_filter(config)
        data = outliers_filter.filter_diffs(data)
        logger.info(f'Finished filtering out outliers after template rendering (rows: {data.shape[0]})')

        logger.info(f'Filtering out diffs with low BLEU score (rows: {data.shape[0]})')
        bleu_filter = get_bleu_filter(config)
        data = bleu_filter.filter_diffs(data)
        logger.info(f'Finished filtering out diffs with low BLEU score  (rows: {data.shape[0]})')

        logger.info(f'Filtering out diffs that remove too much text (rows: {data.shape[0]})')
        removal_filter = RemovalFilter()
        data = removal_filter.filter_diffs(data)
        logger.info(f'Finished filtering out diffs that remove too much text (rows: {data.shape[0]})')

        data.to_csv(ds_cleaned_file, index=False)

    ds_prompts_file = get_tmp_file('ds_prompts')
    if not os.path.exists(ds_prompts_file):
        data = pd.read_csv(ds_cleaned_file, na_values='')
        data.fillna('', inplace=True)

        logger.info('Generating synthetic prompts')
        data = generate_prompts(data, config.prompt_gen, get_tmp_file)
        logger.info('Finished generating synthetic prompts')

        data.to_csv(ds_prompts_file, index=False)

    logger.info('Publishing the dataset')
    data = pd.read_csv(ds_prompts_file, na_values='')
    tmp_dataset_dir = Path(__file__).parent / 'tmp' / config.dataset_type / 'dataset'
    tmp_dataset_dir.mkdir(exist_ok=True)
    create_huggingface_dataset(
        data=data,
        tmp_dir=tmp_dataset_dir,
        dst_dir=config.dst,
        hugging_face_repo_id=config.hf_repo,
        test_size=config.test_size,
    )


def setup_argument_parser():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        type=str,
        help='source directory',
        required=True,
    )
    parser.add_argument(
        '--dst',
        type=str,
        help='destination directory for saving the dataset on disk',
        required=False,
    )
    parser.add_argument(
        '--hf-repo',
        type=str,
        help='HuggingFace Hub repository id for pushing the dataset to the Hub',
        required=False,
    )
    parser.add_argument(
        '--dataset-type',
        '-t',
        type=str,
        choices=['fa', 'npov'],
        help='Dataset type: feature-articles (fa) or npov edits (npov)',
        required=True,
    )
    parser.add_argument(
        '--prompt-gen',
        '-p',
        type=str,
        choices=['simple', 'gpt3'],  # "simple" generation is only for tests; don't use it to create actual dataset
        help='How to generate prompt: simple (hardcoded text for tests) or via gpt-3.5',
        required=False,
        default='gpt3',
    )

    def int_or_float(value):
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f'Invalid value: {value}. Must be an integer or a float.') from e

    parser.add_argument(
        '--test-size',
        type=int_or_float,
        help='proportion of the dataset to include in the test split',
        required=False,
        default=2000,
    )
    return parser


def get_tmp_file_generator(dataset_type: str) -> Callable[[str], Path]:
    """Get function that will return paths to files in tmp directory"""
    dataset_dir = Path(__file__).parent / 'tmp' / dataset_type
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    def get_tmp_file(filename: str) -> Path:
        return dataset_dir / f'{filename}.csv'

    return get_tmp_file


if __name__ == '__main__':
    fix_field_size_limit()
    try:
        main()
    except Exception as e:
        logger.error(f'Failed to create dataset: {e}')
        logger.error(e, exc_info=True)
        exit(1)
