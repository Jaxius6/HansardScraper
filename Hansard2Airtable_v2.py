import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader
import io
import re
from datetime import datetime
import json
import time
from difflib import get_close_matches

# Airtable setup
PAT = 'pat3gAuWKEFFiCZCJ.c19be85e0684f36aaeb153447f2bf233942f49f64efc8edb3d0e90834c717614'
BASE_ID = 'appdeZcAttBaG5oVI'
TABLE_NAME = 'Hansard'
POLITICIANS_TABLE = 'Politicians'  # Add Politicians table name
AIRTABLE_URL = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}'
POLITICIANS_URL = f'https://api.airtable.com/v0/{BASE_ID}/{POLITICIANS_TABLE}'
HEADERS = {
    "Authorization": f"Bearer {PAT}",
    "Content-Type": "application/json"
}

def normalize_name(name):
    """Normalize a name for better matching"""
    # Remove extra spaces and convert to lowercase
    name = ' '.join(name.split()).lower()
    
    # Handle common prefixes consistently
    prefixes = ['hon', 'dr', 'mr', 'mrs', 'ms', 'prof']
    
    # Split name into parts
    parts = name.split()
    
    # Remove prefixes for matching
    if parts and parts[0] in prefixes:
        parts.pop(0)
    
    # Handle 'de', 'van', 'von' etc. consistently
    for i in range(len(parts)-1):
        if parts[i] in ['de', 'van', 'von']:
            parts[i] = parts[i].lower()
    
    return ' '.join(parts)

def find_best_match(name, politicians_map):
    """Find the best matching politician using fuzzy matching"""
    normalized_name = normalize_name(name)
    normalized_politicians = {normalize_name(k): k for k in politicians_map.keys()}
    
    # Debug info
    print(f"\nTrying to match: '{name}'")
    print(f"Normalized to: '{normalized_name}'")
    
    # First try exact match with normalized names
    if normalized_name in normalized_politicians:
        original_name = normalized_politicians[normalized_name]
        print(f"Exact match found: '{original_name}'")
        return politicians_map[original_name]
    
    # Try fuzzy matching
    matches = get_close_matches(normalized_name, normalized_politicians.keys(), n=1, cutoff=0.85)
    if matches:
        original_name = normalized_politicians[matches[0]]
        print(f"Fuzzy matched '{name}' to '{original_name}'")
        return politicians_map[original_name]
    
    return None

def fetch_politicians():
    """
    Fetch all politicians from the Politicians table and create a mapping of names to record IDs
    """
    print("\nFetching politicians from Airtable...")
    politicians_map = {}
    
    try:
        response = requests.get(POLITICIANS_URL, headers=HEADERS)
        response.raise_for_status()
        
        data = response.json()
        records = data.get('records', [])
        print(f"Retrieved {len(records)} politicians")
        
        for record in records:
            full_name = record.get('fields', {}).get('full_name')
            if full_name:
                politicians_map[full_name] = record['id']
                
        return politicians_map
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching politicians: {str(e)}")
        return {}

def process_members(members_string, politicians_map):
    """
    Convert pipe-separated member string into array of Airtable record IDs
    for linked records
    """
    if not members_string:
        return []
        
    # Split by pipe and strip whitespace
    members = [m.strip() for m in members_string.split('|')]
    
    # Process each name and find matching record IDs
    member_ids = []
    unmatched_members = []
    
    for member in members:
        # Remove extra spaces
        member = ' '.join(member.split())
        
        # Handle format "Lastname, Hon Firstname"
        if ',' in member:
            parts = member.split(',', 1)
            lastname = parts[0].strip()
            rest = parts[1].strip()
            # Reconstruct as "Hon Firstname Lastname"
            member = f"{rest} {lastname}"
        
        # First try exact match, then fuzzy match
        member_id = politicians_map.get(member)
        if not member_id:
            member_id = find_best_match(member, politicians_map)
            
        if member_id:
            # Just store the ID
            member_ids.append(member_id)
        else:
            unmatched_members.append(member)
            
    if unmatched_members:
        print(f"\nWarning: Could not find matching records for these members:")
        for member in unmatched_members:
            print(f"  - {member}")
            
    return member_ids  # Return array of IDs

def fetch_existing_records():
    """
    Fetch all existing records from Airtable and create a more comprehensive
    fingerprint for each record to prevent duplications.
    """
    print("\nFetching existing records from Airtable...")
    existing_records = set()
    offset = None
    records_count = 0
    
    try:
        print("Making API request to Airtable...")
        response = requests.get(AIRTABLE_URL, headers=HEADERS)
        response.raise_for_status()
        
        data = response.json()
        records = data.get('records', [])
        print(f"Retrieved {len(records)} records (Total: {len(records)})")
        
        for record in records:
            fields = record.get('fields', {})
            # Convert members list to tuple for hashing
            members = tuple(fields.get('Members', [])) if isinstance(fields.get('Members'), list) else tuple([fields.get('Members')] if fields.get('Members') else [])
            
            fingerprint = (
                fields.get('Date'),
                fields.get('Subject'),
                fields.get('Page'),
                fields.get('House'),
                members,  # Now using the tuple version
                fields.get('Proceeding')
            )
            existing_records.add(fingerprint)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching existing records: {response.status_code} - {response.text}")
        return set()
        
    return existing_records

def add_records_to_airtable(records):
    """
    Add records to Airtable in batches with explicit logging at each step.
    """
    print("\n=== STARTING AIRTABLE UPLOAD PROCESS ===")
    print(f"Attempting to upload {len(records)} records")

    # Debug: Print the first record to verify structure
    if records:
        print("\nFirst record structure:")
        print(json.dumps(records[0], indent=2))

    BATCH_SIZE = 10  # Number of records to send to Airtable at once
    successful_uploads = 0
    failed_uploads = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ---")
        
        # Create payload
        payload = {'records': batch}
        
        try:
            print("Sending request to Airtable...")
            print(f"URL: {AIRTABLE_URL}")
            print("Headers:", {k: '***' if k == 'Authorization' else v for k, v in HEADERS.items()})
            
            # Actually send the request
            response = requests.post(
                AIRTABLE_URL,
                headers=HEADERS,
                json=payload
            )
            
            print(f"\nResponse received from Airtable:")
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Content: {response.text}")
            
            if response.status_code == 200:
                successful_uploads += len(batch)
                print(f"✓ Successfully uploaded batch {i//BATCH_SIZE + 1}")
            else:
                failed_uploads += len(batch)
                print(f"✗ Failed to upload batch {i//BATCH_SIZE + 1}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            failed_uploads += len(batch)
            print(f"✗ Exception occurred during upload: {str(e)}")
            
        print(f"\nCurrent Progress:")
        print(f"Successful uploads so far: {successful_uploads}")
        print(f"Failed uploads so far: {failed_uploads}")
        
        # Small delay between batches
        time.sleep(1)

    print("\n=== UPLOAD PROCESS COMPLETED ===")
    print(f"Final Results:")
    print(f"Total Successful: {successful_uploads}")
    print(f"Total Failed: {failed_uploads}")
    
    return successful_uploads, failed_uploads

# Constants for Airtable limits
AIRTABLE_TEXT_LIMIT = 100000
MAX_TRANSCRIPT_FIELDS = 3  # Number of transcript fields available (Transcript, Transcript2, Transcript3)

def split_transcript(transcript):
    """
    Splits a long transcript across multiple fields intelligently.
    Returns a tuple of (transcripts_dict, was_split, was_truncated)
    """
    if not transcript:
        return {"Transcript": ""}, False, False
        
    total_length = len(transcript)
    if total_length <= AIRTABLE_TEXT_LIMIT:
        return {"Transcript": transcript}, False, False
        
    # Calculate safe limit for each field
    safe_limit = AIRTABLE_TEXT_LIMIT - 100
    transcripts = {}
    remaining_text = transcript
    was_truncated = False
    
    for i in range(MAX_TRANSCRIPT_FIELDS):
        field_name = "Transcript" if i == 0 else f"Transcript{i+1}"
        
        if not remaining_text:
            break
            
        if len(remaining_text) <= safe_limit:
            transcripts[field_name] = remaining_text
            break
            
        # Find a good breaking point
        text_chunk = remaining_text[:safe_limit]
        last_period = text_chunk.rfind('.')
        last_newline = text_chunk.rfind('\n')
        break_point = max(last_period, last_newline) if max(last_period, last_newline) > 0 else safe_limit
        
        # Add the chunk without continuation marker
        current_part = remaining_text[:break_point + 1]
        transcripts[field_name] = current_part
        remaining_text = remaining_text[break_point + 1:]
        
        # If this is the last field and we still have text, mark as truncated
        if i == MAX_TRANSCRIPT_FIELDS - 1 and remaining_text:
            transcripts[field_name] += "\n[Transcript truncated due to length limit]"
            was_truncated = True
            
    return transcripts, len(transcripts) > 1, was_truncated

def create_record(row, member_ids, transcript_fields, existing_records):
    """Create a new record with proper formatting"""
    new_fingerprint = (
        row['Date'],
        row['Subject'],
        row['Page'],
        row['House'],
        tuple(member_ids),
        row['Proceeding']
    )
    
    if new_fingerprint not in existing_records:
        filesize = float(round(row['Filesize'], 2)) if pd.notnull(row['Filesize']) else None
        word_count = len(row['Transcript'].split()) if row['Transcript'] else 0
        
        record = {
            'fields': {
                'Date': row['Date'],
                'Page': row['Page'],
                'Subject': row['Subject'],
                'Proceeding': row['Proceeding'],
                'House': row['House'],
                'Members': member_ids,  # Just the IDs - Airtable will handle the linking
                'PDF': [{'url': row['PDF']}] if row['PDF'] else None,
                'PDF_URL': row['PDF'] if row['PDF'] else None,
                'Word_Count': word_count,
                'Filesize': filesize,
                **transcript_fields
            }
        }
        return record
    return None

# Configuration
ROWS_TO_PROCESS = 100  # Number of rows to scrape from Hansard
BATCH_SIZE = 10  # Number of records to send to Airtable at once

# URL of the Hansard webpage to scrape
target_url = "https://www.parliament.wa.gov.au/hansard/hansard.nsf/NewAdvancedSearch?openform&Query=&Fields=((%5BHan_Member%5D=((%22Walker,%20Hon%20Dr%20Brian%22%20or%20%22Hon%20Dr%20Brian%20Walker%22%20or%20%22Hon%20Dr%20Brian%20Walker%22))))%20and%20((%5BHan_Date%5D%3E=01/01/2021))&sWord=&sWordsSearch=&sWordAll=&sWordExact=&sWordAtLeastOne=&sMember=Walker;%20Hon%20Dr%20Brian&sCommit=&sComms=Current&sHouse=Both%20Houses&sProc=All%20Proceedings&sPage=&sSpeechesFrom=April%202021%20-%20Current&sDateCustom=&sHansardDbs=&sYear=All%20Years&sDate=&sStartDate=&sEndDate=&sParliament=41&sBill=&sWordVar=&sFuzzy=&sResultsPerPage=100&sResultsPage=1&sSortOrd=0&sAdv=1&sRun=true&sContinue=&sWarn="

# Fetch the web page


print("""
                 _ _.-'`-._ _
                ;.'________'.;
     _________n.[____________].n_________
    |""_""_""_""||==||==||==||""_""_""_""]
    |.. .. .. ..||..||..||..||.. .. .. ..|
    |LI LI LI LI||LI||LI||LI||LI LI LI LI|
    |.. .. .. ..||..||..||..||.. .. .. ..|
    |LI LI LI LI||LI||LI||LI||LI LI LI LI|
 ,,;;,;;;,;;;,;;;,;;;,;;;,;;;,;;,;;;,;;;,;;,,
;;;;;;;;;  HANSARD  SCRAPER  ACTIVATED  ;;;;;;
----------------------------------------------

""")


print("Waiting for parliament.wa.gov.au...")
response = requests.get(target_url)
print("URL Success - Scraping Data...")
soup = BeautifulSoup(response.content, 'html.parser')

# Lists to store scraped data
dates, pages, subjects, proceedings, houses, members, pdf_links, transcripts, filesizes, is_truncated = [], [], [], [], [], [], [], [], [], []

# Attempt to locate the table containing Hansard data
tables = soup.find_all('table')
print(f"Found {len(tables)} tables on the page")

if not tables:
    print("No tables found on the page. Please check if the page structure has changed or if JavaScript is loading the content.")
else:
    target_table = None
    for table in tables:
        rows = table.find_all('tr')
        if len(rows) > 1 and 'Date' in rows[0].text:
            target_table = table
            print("Found target table with Hansard data")
            break

    if not target_table:
        print("No suitable table found on the page. Please check the HTML structure.")
    else:
        rows = target_table.find_all('tr')
        print(f"\nProcessing {ROWS_TO_PROCESS} rows from Hansard...")
        
        for idx, row in enumerate(rows[1:ROWS_TO_PROCESS + 1], 1):  # Process specified number of rows
            print(f"\nProcessing row {idx} of {ROWS_TO_PROCESS}...")
            cols = row.find_all('td')
            if len(cols) >= 5:
                date_page = cols[0].text.strip()
                date, page = date_page.split(' / ') if ' / ' in date_page else (date_page, None)
                try:
                    date = datetime.strptime(date, "%d %b %Y").strftime("%Y-%m-%d")
                except ValueError:
                    print(f"Failed to parse date: {date}")
                dates.append(date)
                pages.append(page)
                subject_text = cols[1].text.strip()
                proceeding_match = re.search(r'\[(.*?)\]', subject_text)
                if proceeding_match:
                    proceedings.append(proceeding_match.group(1))
                    subject_text = re.sub(r'\[.*?\]', '', subject_text).strip()
                else:
                    proceedings.append(None)
                subjects.append(subject_text)
                houses.append(cols[2].text.strip())
                members.append(cols[3].text.strip())
                pdf_link = cols[0].find('a', href=True) or cols[4].find('a', href=True)
                if pdf_link:
                    full_pdf_link = "https://www.parliament.wa.gov.au" + pdf_link['href']
                    pdf_links.append(full_pdf_link)
                else:
                    pdf_links.append(None)

                if pdf_link:
                    try:
                        print(f"Downloading PDF from {full_pdf_link}")
                        pdf_response = requests.get(full_pdf_link)
                        filesize = round(len(pdf_response.content) / 1024, 2)  # Convert to KB and round to 2 decimal places
                        filesizes.append(filesize)
                        print(f"PDF size: {filesize} KB")
                        
                        pdf_content = io.BytesIO(pdf_response.content)
                        pdf_reader = PdfReader(pdf_content)
                        print(f"Successfully downloaded PDF with {len(pdf_reader.pages)} pages")
                        transcript_text = ""
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            print(f"Extracting text from page {page_num}/{len(pdf_reader.pages)}")
                            transcript_text += page.extract_text() + "\n"
                        transcripts.append(transcript_text)
                        is_truncated.append(len(transcript_text) > 95000)  # Track if this will need truncation
                    except Exception as e:
                        print(f"Failed to extract text from PDF: {e}")
                        transcripts.append(None)
                        filesizes.append(None)
                        is_truncated.append(False)
                else:
                    print("No PDF link found for this row")
                    transcripts.append(None)
                    filesizes.append(None)
                    is_truncated.append(False)

# Create a DataFrame with the scraped data
hansard_df = pd.DataFrame({
    'Date': dates,
    'Page': pages,
    'Subject': subjects,
    'Proceeding': proceedings,
    'House': houses,
    'Members': members,
    'PDF': pdf_links,
    'Transcript': transcripts,
    'Filesize': filesizes,
    'Truncated': is_truncated
})

print("Parsing Data...")

# Fetch data at startup
print("\nFetching required data...")
politicians_map = fetch_politicians()
existing_records = fetch_existing_records()

# Prepare new records to be added
new_records = []
split_transcripts = []

for _, row in hansard_df.iterrows():
    # Process members into record IDs first
    member_ids = process_members(row['Members'], politicians_map)
    
    # Split transcript if necessary
    transcript_fields, was_split, was_truncated = split_transcript(row['Transcript'])
    if was_split or was_truncated:
        split_transcripts.append({
            'Date': row['Date'],
            'Subject': row['Subject'],
            'Original_Length': len(row['Transcript']),
            'Parts': len(transcript_fields),
            'Was_Truncated': was_truncated
        })
    
    # Create record with proper formatting
    record = create_record(row, member_ids, transcript_fields, existing_records)
    if record:
        new_records.append(record)

# Print detailed duplication and truncation information
print("\nDuplication Check Summary:")
print(f"Total records processed: {len(hansard_df)}")
print(f"New records to add: {len(new_records)}")

if split_transcripts:
    print("\nSplit/Truncated Transcripts:")
    for t in split_transcripts:
        print(f"- {t['Date']}: {t['Subject']}")
        print(f"  Original length: {t['Original_Length']} characters")
        print(f"  Split into: {t['Parts']} parts")
        if t['Was_Truncated']:
            print("  Note: Transcript was truncated")

# Modify the main code to explicitly call and handle the upload results
print("\nPreparing to upload new records...")
if new_records:
    print(f"Found {len(new_records)} new records to upload")
    successful, failed = add_records_to_airtable(new_records)
    print(f"\nFinal Upload Results:")
    print(f"✓ Successfully added {successful} records to Airtable")
    print(f"✗ Failed to add {failed} records")
else:
    print("No new records to upload")

print("Finished Scraping")
