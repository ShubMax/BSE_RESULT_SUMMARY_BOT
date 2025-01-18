import requests
import pandas as pd
from datetime import datetime
import urllib.request
import os
from PyPDF2 import PdfReader
import random
import google.generativeai as genai
import re
import pandas as pd 
from telegram import Bot
from telegram.constants import ParseMode
import asyncio
import os
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import numpy as np
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


if not TELEGRAM_BOT_TOKEN or not GOOGLE_API_KEY:
    raise ValueError("Environment variables not loaded correctly. Check your .env file.")
print("Environment variables loaded successfully.")
print(f"Google API: {GOOGLE_API_KEY}")

bot = Bot(token=TELEGRAM_BOT_TOKEN)

headers = {
        "Connection":
        "keep-alive",
        "Cache-Control":
        "max-age=0",
        "DNT":
        "1",
        "Upgrade-Insecure-Requests":
        "1",
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/111.0.0.0 Safari/537.36",
        "Sec-Fetch-User":
        "?1",
        "Accept":
        "*/*",
        "Sec-Fetch-Site":
        "none",
        "Sec-Fetch-Mode":
        "navigate",
        "Accept-Encoding":
        "gzip, deflate, br",
        "Accept-Language":
        "en-US,en;q=0.9,hi;q=0.8",
        "Referer":
        "https://www.bseindia.com/"
    }   

download_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # App's root directory
PROCESSED_URLS_FILE = os.path.join(BASE_DIR, "processed_urls.txt")
output_folder = os.path.join(BASE_DIR, "images")
os.makedirs(output_folder, exist_ok=True)


# from save_to_image import save_dataframe_images
def save_dataframe_images(df, rows_per_image, output_folder, highlight_dict, title=None, subtitle=None):
    """
    Save a DataFrame as images with specified columns highlighted, including an optional title and subtitle.
    Automatically crops excessive transparent areas from the saved images.

    Args:
        df (pd.DataFrame): DataFrame to save as images.
        rows_per_image (int): Number of rows per image.
        output_folder (str): Folder to save the images.
        highlight_dict (dict): Dictionary where keys are column names and values are highlight colors (e.g., {"Col_2": "#ffcccc"}).
        title (str, optional): Title of the plot. Default is None (no title).
        subtitle (str, optional): Subtitle of the plot. Default is None (no subtitle).
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    
    df = df.reset_index()
    index_col_name = "Metric"  # Rename index for display purposes
    df.rename(columns={"index": index_col_name}, inplace=True)
        # Update highlight dictionary if it includes 'index'
    if 'Metric' not in highlight_dict:
        highlight_dict['Metric'] = "#f0f0f0"  # Light gray for the index column by default
    
    chunks = [df.iloc[i:i + rows_per_image] for i in range(0, len(df), rows_per_image)]

    # Get column indices and their associated colors from the dictionary
    col_indices_colors = {df.columns.get_loc(col): color for col, color in highlight_dict.items() if col in df.columns}


    for idx, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(chunk.shape[1] * 1.4, len(chunk) * 1.8))
        ax.axis('off')

        # Define cell colors (highlight specified columns and handle conditional coloring for growth columns)
        # cell_colors = [["white"] * chunk.shape[1] for _ in range(len(chunk) + 1)]
        cell_colors = [["#d9f2ff"] * chunk.shape[1]]  # Light blue for the header row
        for row_idx, row in enumerate(chunk.itertuples(index=False), start=1):
            row_colors = []
            for col_idx in range(chunk.shape[1]):
                col_name = chunk.columns[col_idx]

                # Conditional formatting for QoQ% and YoY% columns
                if col_name in ["QoQ%", "YoY%"]:
                    cell_value = row[col_idx]
                    if pd.notna(cell_value):  # Check for non-NaN values
                        intensity = min(abs(cell_value) / 100, 1) * 0.5 + 0.5  # Normalize value for intensity (max at 100%)
                        if cell_value > 0:
                            # Green for positive growth
                            row_colors.append((1 - intensity, 1, 1 - intensity, 1))  # Normalized RGBA tuple
                        else:
                            # Red for negative growth
                            row_colors.append((1, 1 - intensity, 1 - intensity, 1))  # Normalized RGBA tuple
                    else :
                        row_colors.append("white") # Default white for NaN values

                # Highlight based on highlight_dict
                elif col_idx in col_indices_colors:
                    row_colors.append(col_indices_colors[col_idx])
                # Default color
                else:
                    row_colors.append("white")
            cell_colors.append(row_colors)

        # Create the table
        table = ax.table(
            cellText=[chunk.columns.tolist()] + chunk.values.tolist(),
            loc='center',
            cellColours=cell_colors,
        ).auto_set_font_size(False)

        # Adjust layout to make space for title and subtitle
        fig.subplots_adjust(top=0.90, bottom=0.05, hspace=0.5)  # Reduced gap between title, table, and subtitle

        # Add title and subtitle if provided
        if title:
            plt.suptitle(title, fontsize=16, weight='bold', y=0.97)  # Title just above the table
        if subtitle:
            plt.figtext(0.5, 0.03, subtitle, ha="center", fontsize=12, style='italic')  # Subtitle just below the table

        # Save the figure temporarily
        temp_path = os.path.join(output_folder, f"temp_image_{idx + 1}.png")
        fig.savefig(
            temp_path,
            bbox_inches='tight', pad_inches=0.0, transparent=True, dpi=300,
        )
        plt.close(fig)

        # Crop the saved image to remove transparent areas
        img = Image.open(temp_path)
        cropped_img = img.crop(img.getbbox())

        # Save the cropped image
        output_path = os.path.join(output_folder, f"df_image_{idx + 1}.png")
        cropped_img.save(output_path)

        # Remove the temporary file
        os.remove(temp_path)

async def send_message_to_telegram(message):
    """
    Send a message to the specified Telegram chat.
    """
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        print(f"Failed to send message to Telegram: {e}")

async def send_images(image_folder: str):
    """
    Sends all images from the specified folder to the user who initiated the Telegram command.
    """
    for image_file in sorted(os.listdir(image_folder)):
        if image_file.endswith(".png"):
            file_path = os.path.join(image_folder, image_file)
            with open(file_path, 'rb') as file:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID,photo=file)
    # Cleanup: Optionally, you may delete the images after sending.
    for file in os.listdir(image_folder):
        os.remove(os.path.join(image_folder, file))

def new_result_url(today_date):
    return f'https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno=1&strCat=Result&strPrevDate={today_date}&strScrip=&strSearch=P&strToDate={today_date}&strType=C&subcategory=-1'

def latest_corp_annoucement(url):
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        print("Response Content:", response.text)
        return pd.DataFrame()  # Return an empty DataFrame on failure

    try:
        data = response.json()
        data_list = data.get('Table', [])
        if not data_list:
            print("No announcements found in the response.")
            return pd.DataFrame()  # Return an empty DataFrame if 'Table' is empty or missing

        df = pd.DataFrame(data_list)
        if 'News_submission_dt' not in df.columns:
            print("The 'News_submission_dt' column is missing from the response.")
            return pd.DataFrame()  # Return an empty DataFrame if the column is missing

        df['ATTACHMENTNAME'] = df['ATTACHMENTNAME'].apply(
            lambda x: f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{x}" if pd.notnull(x) else x
        )
        

        
        data = df.sort_values(by='News_submission_dt', ascending=True)
        data['News_submission_dt'] = pd.to_datetime(data['News_submission_dt'])
        data['date'] = data['News_submission_dt'].dt.date
        data['time'] = data['News_submission_dt'].dt.time
        required_columns = ['SCRIP_CD', 'SLONGNAME', 'CATEGORYNAME', 'SUBCATNAME', 'ATTACHMENTNAME','date','time']
        data = data[required_columns].copy()
        return data

    except Exception as e:
        print(f"Error processing announcements: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on any exception

def load_processed_urls():
    """Load processed URLs from a file."""
    if not os.path.exists(PROCESSED_URLS_FILE):
        return set()
    with open(PROCESSED_URLS_FILE, "r") as file:
        return set(line.strip() for line in file)

def save_processed_url(url):
    """Append a processed URL to the file."""
    with open(PROCESSED_URLS_FILE, "a") as file:
        file.write(url + "\n")

def download_pdf(url):
    try:
        if 'nsearchives.nseindia.com' in url:
            url = url.replace('nsearchives.nseindia.com', 'archives.nseindia.com')

        file_name = url.split("/")[-1]
        temp_dir = "file"  # Temporary directory
        temp_file_path = os.path.join(temp_dir, file_name)
        
        # print(f"Downloading the PDF file from {url}...")
        request = urllib.request.Request(url, headers=download_header)

        with urllib.request.urlopen(request) as response, open(temp_file_path, "wb") as file:
            file.write(response.read())

        # print(f"PDF downloaded and saved as '{file_name}'.")
        return temp_file_path
    except Exception as e:
        print(f"An error occurred while downloading the PDF: {e}")
        return None

def read_pdf_and_extract_text(pdf_file_path):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def get_financial_analysis(text):
    # Hardcoded system instructions
    instructions = (
        "Read the provided financial report and extract the following information in a dataframe format for python : "
        "Columns: Latest Quarter, Previous Quarter, Same Quarter in Previous Year. "
        "Rows: Total Revenue, Total Expenses, Other income ,Profit Befor Tax, Margin (calculated with Total Revenue / Expenses), "
        "Net Profit , EPS . If both standalone and consolidated numbers are available, provide both."
        '''
        write data in this format for consolidated and standalone ( this is an example only) (data should be in millions or cr)
company_name = '' # write company name here 

data_standalone = {

    'Total Revenue': {
        'Latest Quarter': 5543.6, #Converted to lakhs
        'Previous Quarter': 4577.1, #Converted to lakhs
        'Same Quarter in Previous Year': 3678.0, #Converted to lakhs
    },
    'Total Expenses': {
        'Latest Quarter': 4866.2, #Converted to lakhs
        'Previous Quarter': 3981.5, #Converted to lakhs
        'Same Quarter in Previous Year': 3498.2, #Converted to lakhs

    },
    'other income': {
        'Latest Quarter': 5453.1,  #Converted to lakhs
        'Previous Quarter': 4459.3, #Converted to lakhs
        'Same Quarter in Previous Year': 3653.0, #Converted to lakhs

    },
    'PBT': {
        'Latest Quarter': 677.4, #Converted to lakhs
        'Previous Quarter': 595.5, #Converted to lakhs
        'Same Quarter in Previous Year': 179.8, #Converted to lakhs

    },
    'Margin': {
        'Latest Quarter': (5453.1 - 4866.2) / 5453.1 * 100,
        'Previous Quarter': (4459.3 - 3981.5) / 4459.3 * 100,
        'Same Quarter in Previous Year': (3653.0 - 3498.2) / 3653.0 * 100,

    },
    'Net Profit': {
        'Latest Quarter': 505.1, #Converted to lakhs
        'Previous Quarter': 421.0, #Converted to lakhs
        'Same Quarter in Previous Year': 134.0, #Converted to lakhs

    }
    'EPS': {
        'Latest Quarter': 23.1, 
        'Previous Quarter': 24.0, 
        'Same Quarter in Previous Year': 22.0

    }
}
summary = # write summary about latest quaterly number  here and any promoter commentary or any note in that file
        '''
    )

    return [instructions,text]

def generate_summary_with_gemini(text):
    models = ["gemini-1.5-flash", "gemini-1.5-pro"] # 'gemini-1.5-flash-8b' , gemini-2.0-flash-exp'
    model = random.choice(models)
    # model = 'gemini-2.0-flash-exp'

    generation_config = {
        "response_mime_type": "text/plain",
        "max_output_tokens": 8000,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name=model,generation_config=generation_config,)
        count_tokens = model.count_tokens(text)
        response = model.generate_content(text)
        print("Required Tokens : ",count_tokens)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error with Gemini API: {e}")
        return None

def df_outlines(df):
    # Determine column widths
    col_widths = [max(len(str(x)) for x in [col] + df[col].tolist()) for col in df.columns]
    index_width = max(len(str(x)) for x in df.index) + 2
    
    # Create the horizontal line
    line = "+" + "+".join("-" * (w + 2) for w in [index_width] + col_widths) + "+"

    # Build the table
    rows = [line]
    # Header row
    header = "|" + f" {'Index'.ljust(index_width)} " + "|" + "|".join(
        f" {col.ljust(w)} " for col, w in zip(df.columns, col_widths)
    ) + "|"
    rows.append(header)
    rows.append(line)
    # Data rows with row outlines
    for idx, row in df.iterrows():
        row_data = "|" + f" {str(idx).ljust(index_width)} " + "|" + "|".join(
            f" {str(val).rjust(w)} " for val, w in zip(row, col_widths)
        ) + "|"
        rows.append(row_data)
        rows.append(line)  # Add a line after each row
    
    return "\n".join(rows)

def center_text(message):
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Calculate padding for the text to be centered with borders
    total_padding = terminal_width - len(message) - 4  # Subtract length of message and borders
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    formatted_message = ("-" * left_padding + " " + message + " " + "-" * right_padding )
    print(formatted_message)

async def process_announcements():

    def extract_dictionary(code_str, dict_name):
        """
        Extract a dictionary block by detecting matching braces for the given variable name.
        """
        # Match the start of the dictionary by name
        pattern = rf"{dict_name}\s*=\s*\{{"
        match = re.search(pattern, code_str)
        if not match:
            return None

        # Start of the dictionary
        start_idx = match.start() + len(match.group()) - 1
        open_braces = 1  # We've matched the initial brace

        # Find the corresponding closing brace
        for idx in range(start_idx + 1, len(code_str)):
            if code_str[idx] == '{':
                open_braces += 1
            elif code_str[idx] == '}':
                open_braces -= 1

            # All braces are matched
            if open_braces == 0:
                extracted = code_str[match.start():idx + 1]
                # Remove the variable assignment part to isolate the dictionary
                return extracted.split('=', 1)[-1].strip()

        # Return None if no match was found
        return None

    def parse_to_dict(dictionary_string):
        """
        Safely parses the extracted dictionary string into a Python dictionary.
        """
        if dictionary_string is None:
            return None
        try:
            return eval(dictionary_string)
        except Exception as e:
            print(f"Error parsing dictionary: {e}")
            return None

    processed_urls = load_processed_urls()
    while True:
        today = datetime.now().strftime("%Y%m%d")
        result_url = new_result_url(today)
        announcements = latest_corp_annoucement(result_url)

        if announcements.empty:
            print("No new announcements. Waiting...")
        else:
            for _, row in announcements.iterrows():
                attachment_url = row['ATTACHMENTNAME']
                if attachment_url and attachment_url not in processed_urls:
                    processed_urls.add(attachment_url)
                    save_processed_url(attachment_url)
                    file_name = download_pdf(attachment_url)
                    if file_name:
                        pdf_text = read_pdf_and_extract_text(file_name)
                        if pdf_text.strip():
                            prompt = get_financial_analysis(pdf_text)
                            ai_response = generate_summary_with_gemini(prompt)
                            if ai_response:
                                print(f"\n--- Analysis for {row['SLONGNAME']} : {file_name} ---\n")
                                print(f"{row['SLONGNAME']}     Time : {row['date']}-{row['time']}")
                                company_info = (
                                    f"*{row['SLONGNAME']}*\n"
                                    f"_Time: {row['date']} - {row['time']}_\n"
                                    f"_Link: {row['ATTACHMENTNAME']}_"
                                )
                                await send_message_to_telegram(company_info)
                                
                                standalone_string = extract_dictionary(ai_response, "data_standalone")
                                standalone_dict = parse_to_dict(standalone_string)

                                if standalone_dict:
                                    standalone_df = pd.DataFrame(standalone_dict).T.round(2)
                                    standalone_df['QoQ%'] = np.divide(
                                        (standalone_df['Latest Quarter'] - standalone_df['Previous Quarter']) * 100,
                                        standalone_df['Previous Quarter'].abs(),
                                        out=np.full_like(standalone_df['Latest Quarter'], np.nan),  # Default value for invalid operations
                                        where=standalone_df['Previous Quarter'].abs() != 0         # Avoid division by zero
                                    ).round(2)
                                    standalone_df['YoY%'] = np.divide(
                                        (standalone_df['Latest Quarter'] - standalone_df['Same Quarter in Previous Year']) * 100,
                                        standalone_df['Same Quarter in Previous Year'].abs(),
                                        out=np.full_like(standalone_df['Latest Quarter'], np.nan),  # Default value for invalid operations
                                        where=standalone_df['Same Quarter in Previous Year'].abs() != 0  # Avoid division by zero
                                    ).round(2)
                                    standalone_df['QoQ%'] = standalone_df['QoQ%'].replace([np.inf], 9999).replace([-np.inf], -9999)
                                    standalone_df['YoY%'] = standalone_df['YoY%'].replace([np.inf], 9999).replace([-np.inf], -9999)
                                    standalone_df.rename(columns={"Same Quarter in Previous Year": "2023 Q3"}, inplace=True)
                                    standalone_df.rename(columns={"Previous Quarter": "2024 Q2"}, inplace=True)
                                    standalone_df.rename(columns={"Latest Quarter": "2024 Q3"}, inplace=True)
                                    standalone_df.fillna(0,inplace = True)
                                    center_text("Standalone Financial Statement")
                                    print(df_outlines(standalone_df))                                    
                                    highlight_columns = {'2024 Q3' : '#95c2bf','2024 Q2':'#cedbde','2023 Q3':'#95c2bf'}
                                    save_dataframe_images(standalone_df,rows_per_image=10,output_folder=output_folder,highlight_dict=highlight_columns)
                                    await send_images(output_folder)
                                else:
                                    print(f"{row["SLONGNAME"]} Standalone Financial Statement Not available or Not able to fetch.")
                                    await send_message_to_telegram(f"{row["SLONGNAME"]} Standalone Financial Statement Not available or Not able to fetch.")

                                # Extract and parse the "data_consolidated" dictionary
                                consolidated_string = extract_dictionary(ai_response, "data_consolidated")
                                consolidated_dict = parse_to_dict(consolidated_string)

                                if consolidated_dict:
                                    consolidated_df = pd.DataFrame(consolidated_dict).T.round(2)
                                    consolidated_df['QoQ%'] = np.divide(
                                        (consolidated_df['Latest Quarter'] - consolidated_df['Previous Quarter']) * 100,
                                        consolidated_df['Previous Quarter'].abs(),
                                        out=np.full_like(consolidated_df['Latest Quarter'], np.nan),  # Default value for invalid operations
                                        where=consolidated_df['Previous Quarter'].abs() != 0         # Avoid division by zero
                                    ).round(2)
                                    consolidated_df['YoY%'] = np.divide(
                                        (consolidated_df['Latest Quarter'] - consolidated_df['Same Quarter in Previous Year']) * 100,
                                        consolidated_df['Same Quarter in Previous Year'].abs(),
                                        out=np.full_like(consolidated_df['Latest Quarter'], np.nan),  # Default value for invalid operations
                                        where=consolidated_df['Same Quarter in Previous Year'].abs() != 0  # Avoid division by zero
                                    ).round(2)
                                    consolidated_df['QoQ%'] = consolidated_df['QoQ%'].replace([np.inf], 9999).replace([-np.inf], -9999)
                                    consolidated_df['YoY%'] = consolidated_df['YoY%'].replace([np.inf], 9999).replace([-np.inf], -9999)
                                    consolidated_df.rename(columns={"Same Quarter in Previous Year": "2023 Q3"}, inplace=True)
                                    consolidated_df.rename(columns={"Previous Quarter": "2024 Q2"}, inplace=True)
                                    consolidated_df.rename(columns={"Latest Quarter": "2024 Q3"}, inplace=True)
                                    consolidated_df.fillna(0,inplace = True)
                                    center_text("Consolidated Financial Statement")
                                    print(df_outlines(consolidated_df))
                                    highlight_columns = {'2024 Q3' : '#95c2bf','2024 Q2':'#cedbde','2023 Q3':'#95c2bf'}
                                    save_dataframe_images(consolidated_df,rows_per_image=10,output_folder=output_folder,highlight_dict=highlight_columns)
                                    await send_images(output_folder)
                                else:
                                    print(f"{row["SLONGNAME"]} Consolidated Financial Statement Not available or Not able to fetch.")
                                    await send_message_to_telegram(f"{row["SLONGNAME"]} Consolidated Financial Statement Not available or Not able to fetch.")
                            else:
                                print(f"Failed to generate summary for {file_name}.")
                        os.remove(file_name)

        await asyncio.sleep(120)  # Wait for 10 minutes before checking again

def main():
    asyncio.run(process_announcements())

if __name__ == "__main__":
    main()
