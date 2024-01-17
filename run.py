import asyncio
import aiohttp
import config
import csv
import json
import os
import pandas as pd
import sys
import time
import re
import sys
from base64 import b64decode
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class ProgressLog:
    def __init__(self, total):
        self.total = total
        self.done = 0

    def increment(self):
        self.done = self.done + 1

    def __repr__(self):
        return f"Done runs {self.done}/{self.total}."


class ParseErrorLog:
    def __init__(self, total):
        self.total = total
        self.errors = 0

    def increment(self):
        self.errors = self.errors + 1

    def __repr__(self):
        return f"Errors {self.errors}/{self.total}."


def fetch_keyterms(file):
    data = pd.read_excel(file)
    kt = data.iloc[0:, 0:].values.tolist()
    return kt


# Create Detail File
def create_output_file():
    header = ['index', 'gpt_prompt', 'image_prompt', 'image_gen_status', 'revised_prompt', 'drive_link']
    with open(CSV_OUTPUT_DETAIL, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)


# Write to the CSV output files
def write_to_csv(filename, output_data):
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(output_data)


# Callback when can retries are exhausted
def all_retries_failed(retry_state):
    print('All retries failed!')
    time.sleep(65)
    ## Log detail sheet
    detail_temp_data = ['', '', '']
    with open(CSV_OUTPUT_DETAIL, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(detail_temp_data)


# System Pause
async def pause():
    rate_limit_event.clear() # Rate limit is likely hit, or network error, so we cooldown for a bit
    print('Rate limit hit, pausing execution')
    await asyncio.sleep(60)
    rate_limit_event.set() # 60 second window has passed, clear the rate-limit indicator to resume operations
    print('Flag cleared, resuming...')


# Asyncio Request Run
@retry(wait=wait_random_exponential(min=60, max=125), stop=stop_after_attempt(5), before_sleep=print, retry_error_callback=all_retries_failed)
async def get_completion(index, user_prompt, session, semaphore, progress_log, error_log):
    # Wait for the event to be set before attempting to proceed
    await rate_limit_event.wait()
    async with semaphore:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, 
                                json={
                                    "model": "gpt-4-0613",
                                    "messages": [
                                            {"role": "user", "content": user_prompt}
                                        ],
                                    "temperature": 1,
                                    "tools": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "generate_image",
                                                "description": "Generates an image",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "imagePrompt": {
                                                            "type": "string",
                                                            "description": "A prompt used to generate the image."
                                                        }
                                                    },
                                                    "required": ["imagePrompt"]
                                                }
                                            }
                                        }
                                    ],
                                    "tool_choice": {
                                        "type": "function", 
                                        "function": {"name": "generate_image"}
                                    }
                                }) as resp:

            if resp.status == 429:
                await pause()
                raise Exception("Rate limit was hit, request should now be retried.") # Re-raising to be retried

            try:
                response_json = await resp.json()
                completion_text = response_json["choices"][0]['message']["tool_calls"][0]["function"]["arguments"]
                # print(response_json)

            except Exception as e:
                raise # Re-raising to be retried

            else:
                # Parse the completion_text
                try:
                    completion_obj = completion_text.strip()
                    completion_obj = completion_obj[completion_obj.find('{'):completion_obj.find('}')+1]
                    completion_obj = json.loads(completion_obj)
                    imagePrompt = completion_obj.get('imagePrompt', '')

                except Exception as e:
                    print(f'\tParsing Error for imagePrompt: {index} | Completion object: ', end='')
                    print(completion_text.strip())
                    completion_obj = {}
                    imagePrompt = ''

                    # Log error count
                    error_log.increment()
                    print(error_log)

                    print('\tRetrying...')
                    raise # Re-raising to be retried

                else:
                    # Log successful execution (prompt) count
                    progress_log.increment()
                    print(progress_log)

                    # Generate Image and Upload to Drive
                    if (imagePrompt != ''):
                        async with session.post("https://api.openai.com/v1/images/generations", headers=HEADERS, 
                                json={
                                    "model": "dall-e-3",
                                    "quality": "hd",
                                    "size": "1024x1792",
                                    "response_format": "b64_json",
                                    "prompt": imagePrompt
                                }) as img_resp:

                            img_resp_status = img_resp.status
                            revised_prompt = ''
                            drive_image_link = ''

                            if (img_resp_status == 400):
                                drive_image_link = "content policy violation error"
                                img_response_json = await img_resp.json()
                                img_resp_status = f'{img_resp_status} - {img_response_json.get("error", {}).get("message", "")}'

                            elif (img_resp_status == 200):
                                drive_image_link = ''
                                try:
                                    # Decode b64 encoded data to a png file
                                    img_response_json = await img_resp.json()
                                    image_data = img_response_json.get('data', [{}])
                                    b64_json =  image_data[0].get('b64_json', '')
                                    revised_prompt = image_data[0].get('revised_prompt', '')
                                    image_data = b64decode(b64_json)

                                    # Save Image locally
                                    image_name = f'{index}_{RANDOMIZER}.png'
                                    image_file_path = os.path.join(IMAGE_OUTPUT_DIR, image_name)
                                    with open(image_file_path, mode="wb") as png:
                                        png.write(image_data)

                                except Exception as e:
                                    print(f'\t{str(e)}: Error saving b64 content')
                                    raise

                                else:
                                    try:
                                        # Upload to Drive
                                        file_metadata = {
                                            'name': image_name,
                                            'parents': [DRIVE_FOLDER_ID]
                                        }
                                        media = MediaFileUpload(image_file_path)
                                        file = DRIVE_SERVICE.files().create(
                                            body=file_metadata, 
                                            media_body=media, 
                                            fields='id').execute()
                                        file_id = file.get('id')
                                        drive_image_link = f"https://drive.google.com/file/d/{file_id}/view"
                                    
                                    except Exception as e:
                                        print(f'\t{str(e)}: Error uploading image to Drive')
                                        raise

                            elif (img_resp_status == 429):
                                await pause()
                                raise Exception("Rate limit was hit while generating images, request should now be retried.") # Re-raising to be retried
                            
                            else:
                                drive_image_link = 'Network error while generating image'

                    else:
                        img_resp_status = 'NA: Image prompt not generated'

                    # Add to detail sheet
                    detail_temp_data = [index, user_prompt, imagePrompt, img_resp_status, revised_prompt, drive_image_link]
                    with open(CSV_OUTPUT_DETAIL, 'a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(detail_temp_data)
                    
            return ''


  

async def get_completion_list(content_list, max_parallel_calls, timeout):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(content_list))
    error_log = ParseErrorLog(len(content_list))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
        return await asyncio.gather(*[get_completion(content[0], content[1], session, semaphore, progress_log, error_log) for content in content_list])


async def main():

    # Load Content
    try:
        os.makedirs(IMAGE_OUTPUT_DIR)
    except FileExistsError:
        print('Output directory already exists.\n')
    finally:
        create_output_file()
        content_list = fetch_keyterms(EXCEL_INPUT)
        max_parallel_calls = 10
        timeout = 60

        # Run
        responses = await get_completion_list(content_list, max_parallel_calls, timeout)


# Read OpenAI token file
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
    "OpenAI-Organization": config.ORG_ID
}

# Setup other important constants
IMAGE_OUTPUT_DIR = config.IMAGE_OUTPUT_DIR
EXCEL_INPUT = config.EXCEL_INPUT
RANDOMIZER = int(time.time())
CSV_OUTPUT_DETAIL = f"imagePrompts_{RANDOMIZER}.csv"

# Google Drive Setup
CREDS_PATH = config.CREDS_PATH
DRIVE_FOLDER_ID = config.DRIVE_FOLDER_ID
DRIVE_CREDS = Credentials.from_service_account_file(CREDS_PATH, scopes=['https://www.googleapis.com/auth/drive'])
DRIVE_SERVICE = build('drive', 'v3', credentials=DRIVE_CREDS)

## Run
rate_limit_event = asyncio.Event()
rate_limit_event.set()  # Initially not hit, operations contd.
asyncio.run(main())
