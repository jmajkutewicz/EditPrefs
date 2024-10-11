import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from wikiprefs.collect_retained_diffs import DiffRow
from wikiprefs.create_dataset import PromptGenerationException, generate_prompts


def create_diff_row(title, section, new_text):
    row = DiffRow(
        page_title=title,
        page_id=1,
        rev_id=3,
        prev_rev_id=2,
        timestamp='5.5.2005',
        contributor='x',
        comment='removed text',
        section=section,
        old_text='',
        new_text=new_text,
    )
    return row


def _create_generator_mock(mock_gpt3_prompt_generator_class):
    generator_instance = MagicMock()
    mock_gpt3_prompt_generator_class.return_value = generator_instance
    return generator_instance


class TestGeneratePromptsWithGpt3(unittest.TestCase):

    def test_gpt3_prompts_generation(self):
        self.skipTest('Manual test')

        with tempfile.TemporaryDirectory() as tmpdir_name:
            # given
            tmp_file = os.path.join(tmpdir_name, 'test_prompts.csv')

            diffs = [
                # wrong section, remaining <ref> content
                create_diff_row('Bob Dylan', 'Life and career // 1960s // Relocation to New York and record deal',
                                'On November 4, 1971 Dylan recorded "George Jackson" which he released a week later.For many, the single was a surprising return to protest material, mourning the killing of Black Panther George Jackson in San Quentin Prison that summer.Gray, The Bob Dylan Encyclopedia, ´pp.342–343.'),
                create_diff_row('Bob Dylan', 'Life and career // 1960s // Relocation to New York and record deal',
                                'Family\nDylan married Sara Lownds on November 22, 1965.Their first child, Jesse Byron Dylan, was born on January 6, 1966, and they had three more children: Anna Lea (born July 11, 1967), Samuel Isaac Abram (born July 30, 1968), and Jakob Luke (born December 9, 1969).Dylan also adopted Sara\'s daughter from a prior marriage, Maria Lownds (later Dylan, born October 21, 1961).Bob and Sara Dylan were divorced on June 29, 1977.Gray (2006), pp.198–200.He told Kurt Loder of "Rolling Stone" magazine: "I\'ve never said I\'m born again.That\'s just a media term.I don\'t think I\'ve been an agnostic.I\'ve always thought there\'s a superior power, that this is not the real world and that there\'s a world to come.'),
                create_diff_row('Shakespeare authorship question', 'efault__wiki__article__introduction__heading',
                                'In the  19th Century the most popular alternative candidate was Sir Francis Bacon.Many 19th century doubters, however, declared themselves agnostics and refused to endorse an alternative.The American populist poet Walt Whitman gave voice to this popular skepticism when he told Horace Traubel,  "I go with you fellows when you say no to Shaksper: that\'s about as far as I have got.As to Bacon, well, we\'ll see, we\'ll see."'),
                create_diff_row('No. 1 Flying Training School RAAF', 'History // World War II',
                                'No. 1 SFTS came under the control of Southern Area Command, headquartered in Melbourne.'),
                create_diff_row('Three-cent nickel', 'Legislation',
                                'Q. David Bowers said of the sudden passage of the legislation "We can only guess what happened behind the scenes".'),
                create_diff_row('Year Zero (album)', 'Personnel', 'Credits adapted from the CD liner notes.\n\n* Trent Reznor – vocals, writer, instrumentation, production, engineering and art direction'),
                create_diff_row('Halo Wars', 'default__wiki__article__introduction__heading',
                                '"Halo Wars" is a real-time strategy (RTS) video game developed by Ensemble Studios and published by Microsoft Game Studios for the Xbox 360 video game console.'),
                create_diff_row('Planet Stories', 'Bibliographic details',
                                'The editorial succession at "Planet" was:'),
                create_diff_row('Manta ray', 'Conservation issues', 'The greatest threat to manta rays is overfishing.'),
                create_diff_row('Ich will den Kreuzstab gerne tragen, BWV 56', 'Music // Movements // 3',
                                ': Da krieg ich in dem Herren Kraft,\n: Da hab ich Adlers Eigenschaft,\n: Da fahr ich auf von dieser Erden\n: Und laufe sonder matt zu werden.\n: O gescheh es heute noch!\n|Joyful, joyful now am I,\nFor the yoke is light upon me.'),
                create_diff_row('Peregrine falcon', 'default__wiki__article__introduction__heading',
                                'The Peregrine\s breeding range includes land regions from the Arctic tundra to the Tropics.It can be found nearly everywhere on Earth, except extreme polar regions, very high mountains, and most tropical rainforests; the only major ice-free landmass from which it is entirely absent is New Zealand.This makes it the world\'s most widespread bird of prey.Both the English and scientific names of this species mean "wandering falcon", referring to the migratory habits of many northern populations.'),
                create_diff_row('Elk', 'default__wiki__article__introduction__heading',
                                'Elk range in forest and forest-edge habitat, feeding on grasses, plants, leaves, and bark.'),
            ]
            df = pd.DataFrame([p.__dict__ for p in diffs])

            generate_prompts(df, 'gpt3', lambda _: tmp_file)

            df.to_csv('resources/gpt3_prompts.csv')

            # no assertions, manually check if GPT-3 generated viable questions
            for i, item in df.iterrows():
                q = item['prompt']
                a = item['new_text']
                print('=======================\n')
                print(f'Q:{q}:\n\nA:{a}\n')

    @patch('wikiprefs.prompts.gpt3.Gpt3PromptGenerator')
    def test_prompt_generation(self, mock_gpt3_prompt_generator_class):
        with tempfile.TemporaryDirectory() as tmpdir_name:
            # given
            tmp_file = os.path.join(tmpdir_name, 'test_prompts.csv')

            mock_prompt = 'prompt test'
            generator_instance = _create_generator_mock(mock_gpt3_prompt_generator_class)
            generator_instance.generate_prompt.return_value = mock_prompt

            df = pd.DataFrame([create_diff_row('page', 'section', 'text').__dict__])

            # when
            df = generate_prompts(df, 'gpt3', lambda _: tmp_file)

            # then
            self.assertEqual(df.at[0, 'prompt'], mock_prompt)
            mock_gpt3_prompt_generator_class.assert_called_once()
            generator_instance.generate_prompt.assert_called()

    @patch('wikiprefs.prompts.gpt3.Gpt3PromptGenerator')
    def test_prompt_generation_retry(self, mock_gpt3_prompt_generator_class):
        with tempfile.TemporaryDirectory() as tmpdir_name:
            # given
            tmp_file = os.path.join(tmpdir_name, 'test_prompts.csv')

            mock_prompt = 'prompt test'
            def generate_prompt(page_title, section, text):
                if page_title == 'page2':
                    raise Exception('Test')
                return mock_prompt
            generator_instance = _create_generator_mock(mock_gpt3_prompt_generator_class)
            generator_instance.generate_prompt.side_effect = generate_prompt

            df = pd.DataFrame([
                create_diff_row('page', 'section', 'text').__dict__,
                create_diff_row('page2', 'section', 'text').__dict__,
            ])

            # when
            try:
                df = generate_prompts(df, 'gpt3', lambda _: tmp_file)
            except PromptGenerationException:
                pass
            else:
                self.fail('Prompt generation exception was expected')

            # then
            mock_gpt3_prompt_generator_class.assert_called_once()
            self.assertEqual(generator_instance.generate_prompt.call_count, 2)
            self.assertEqual(df.at[0, 'prompt'], mock_prompt)
            self.assertEqual(df.at[1, 'prompt'], '')

            # retry
            mock_prompt_2 = 'prompt 2'
            def generate_prompt(page_title, section, text):
                return mock_prompt_2
            generator_instance.generate_prompt.side_effect = generate_prompt

            df = generate_prompts(df, 'gpt3', lambda _: tmp_file)

            self.assertEqual(generator_instance.generate_prompt.call_count, 3)
            self.assertEqual(df.at[0, 'prompt'], mock_prompt)
            self.assertEqual(df.at[1, 'prompt'], mock_prompt_2)

if __name__ == '__main__':
    unittest.main()
