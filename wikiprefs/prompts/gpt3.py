import json
import multiprocessing
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

logger = multiprocessing.get_logger()


class Gpt3PromptGenerator:
    """Generates prompts (i.e. instruction/question) for given text (i.e. answers) using OpenAI GPT3"""

    def __init__(self):
        """Initialize the Gpt3PromptGenerator"""
        self.client = OpenAI()

        # load the template
        resources_path = os.path.join(Path(__file__).parent.parent, 'resources')
        env = Environment(loader=FileSystemLoader(resources_path), autoescape=False)
        self.template = env.get_template('gpt3-template.j2')

        self.instruction_prefix = 'Instruction:'
        self.instruction_prefix_len = len(self.instruction_prefix)

    def _get_messages(self, page_title, section, text):
        """Get message for GPT as json string"""
        # JSON encode the inputs to ensure proper escaping
        page_title_json = json.dumps(page_title)[1:-1]  # Strip the outer quotes
        section_json = json.dumps(section)[1:-1]
        text_json = json.dumps(text)[1:-1]

        # Render prompt with JSON-safe strings
        template_data = {
            'article': page_title_json,
            'section': section_json,
            'text': text_json,
        }
        output = self.template.render(template_data)

        try:
            messages = json.loads(output)
            return messages
        except json.JSONDecodeError as e:
            logger.error(f'Failed to decode JSON: {e}')
            return None

    def generate_prompt(self, page_title: str, section: str, new_text: str) -> str:
        """Generate prompt for given text

        This will send a request to GPT3 to generate a prompt/instruction/question
        to which answer will be the given text
        """
        messages = self._get_messages(page_title, section, new_text)

        completion = self.client.chat.completions.create(model='gpt-3.5-turbo', messages=messages)

        prompt = completion.choices[0].message
        logger.debug(prompt)

        prompt = prompt.content
        if prompt.startswith(self.instruction_prefix):
            # GPT3 sometimes (often?) start the response with repeated "Instruction:" from the prompt. Try to remove it
            logger.debug(f'Stripping "Instruction:" prefix from prompt: {prompt}')
            prompt = prompt[self.instruction_prefix_len :].lstrip()

        return prompt
