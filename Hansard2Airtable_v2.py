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
    if not name:
        return ""
    
    # Remove any non-printable characters and normalize spaces
    name = ''.join(char for char in name if char.isprintable())
    name = ' '.join(name.split()).strip()
    
    # Handle "Lastname, Title Firstname" format
    if ',' in name:
        parts = name.split(',', 1)
        lastname = parts[0].strip()
        rest = parts[1].strip()
        
        # Split rest into title and firstname
        rest_parts = rest.split(None, 1)
        if len(rest_parts) > 1 and rest_parts[0].lower() in ['mr', 'ms', 'mrs', 'hon', 'dr']:
            title = rest_parts[0]
            firstname = rest_parts[1]
            name = f"{title} {firstname} {lastname}"
        else:
            name = f"{rest} {lastname}"
    
    return name.strip()

def normalize_name_for_matching(name):
    """Normalize a name for fuzzy matching comparison"""
    if not name:
        return ""
    
    # Remove any non-printable characters and normalize spaces
    name = ''.join(char for char in name if char.isprintable())
    name = ' '.join(name.split()).strip()
    
    # Remove common titles for matching
    titles = ['hon', 'dr', 'mr', 'mrs', 'ms', 'prof']
    words = name.lower().split()
    if words and words[0] in titles:
        words.pop(0)
    
    return ' '.join(words)

def find_best_match(name, politicians_map):
    """Find the best matching politician using fuzzy matching"""
    if not name:
        return None
        
    # First normalize both strings for comparison
    normalized_name = normalize_name_for_matching(name)
    normalized_politicians = {normalize_name_for_matching(k): k for k in politicians_map.keys()}
    
    # Debug info
    print(f"\nTrying to match: '{name}'")
    print(f"Normalized to: '{normalized_name}'")
    
    # Try exact match with normalized names
    for norm_pol, original_pol in normalized_politicians.items():
        if normalized_name == norm_pol:
            print(f"Found exact match: '{original_pol}'")
            return politicians_map[original_pol]
    
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
    global stats
    
    if not members_string or members_string.isspace():
        stats['empty_members'] += 1
        stats['invalid_members'].add(members_string)  # Add the empty/invalid string
        return []
    
    # Split by pipe and strip whitespace
    members = [m.strip() for m in members_string.split('|')]
    members = [m for m in members if m and not m.isspace()]  # Remove empty or whitespace-only entries
    
    if not members:  # If after cleaning we have no valid members
        stats['empty_members'] += 1
        return []
    
    # Process each name and find matching record IDs
    member_ids = []
    unmatched_members = []
    
    for member in members:
        # Skip certain roles that aren't actual members
        if member.upper() in ['ACTING SPEAKER', 'SPEAKER', 'PRESIDENT', 'DEPUTY SPEAKER']:
            stats['invalid_members'].add(member)
            continue
            
        # Normalize the name
        normalized_name = normalize_name(member)
        print(f"\nProcessing member: '{member}'")
        print(f"Normalized to: '{normalized_name}'")
        
        # First try exact match with normalized name
        member_id = politicians_map.get(normalized_name)
        if member_id:
            print(f"Found exact match: '{normalized_name}'")
            member_ids.append(member_id)
            continue
            
        # Try fuzzy match
        member_id = find_best_match(normalized_name, politicians_map)
        if member_id:
            member_ids.append(member_id)
        else:
            unmatched_members.append(member)
            stats['unmatched_members'].add(member)
            print(f"Added to unmatched members: {member}")  # Debug output
            
    if unmatched_members:
        print(f"\nWarning: Could not find matching records for these members:")
        for member in unmatched_members:
            print(f"  - {member}")
            
    return member_ids

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
                tuple(fields.get('Proceeding', []))  # Use tuple for proceedings
            )
            existing_records.add(fingerprint)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching existing records: {response.status_code} - {response.text}")
        return set()
        
    return existing_records

def upload_records_to_airtable(records):
    """Upload records to Airtable in batches"""
    if not records:
        print("No new records to upload.")
        return
    
    BATCH_SIZE = 10  # Number of records to send to Airtable at once
    successful_uploads = 0
    failed_uploads = 0
    
    # Split records into batches
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num, i in enumerate(range(0, len(records), BATCH_SIZE), 1):
        batch = records[i:i + BATCH_SIZE]
        print(f"\n--- Processing Batch {batch_num}/{total_batches} ---")
        
        # Prepare the request
        url = f"https://api.airtable.com/v0/{BASE_ID}/Hansard"
        headers = {
            'Authorization': f'Bearer {PAT}',
            'Content-Type': 'application/json'
        }
        data = {'records': batch}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            # Only print essential response info
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                successful_uploads += len(batch)
                print(f"✓ Successfully uploaded {len(batch)} records")
            else:
                failed_uploads += len(batch)
                print(f"✗ Failed to upload batch {batch_num}")
                print(f"Error: {response.json().get('error', {}).get('message', 'Unknown error')}")
            
        except Exception as e:
            failed_uploads += len(batch)
            print(f"✗ Exception during upload: {str(e)}")
        
        print(f"\nCurrent Progress:")
        print(f"Successful uploads so far: {successful_uploads}")
        print(f"Failed uploads so far: {failed_uploads}")
        
    return successful_uploads, failed_uploads

def split_transcript(transcript):
    """
    Splits a long transcript across multiple fields intelligently.
    Returns a tuple of (transcripts_dict, was_split, was_truncated)
    """
    if not transcript:
        return {"Transcript": ""}, False, False
        
    total_length = len(transcript)
    if total_length <= 100000:
        return {"Transcript": transcript}, False, False
        
    # Calculate safe limit for each field
    safe_limit = 100000 - 100
    transcripts = {}
    remaining_text = transcript
    was_truncated = False
    
    for i in range(3):
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
        if i == 2 and remaining_text:
            transcripts[field_name] += "\n[Transcript truncated due to length limit]"
            was_truncated = True
            
    return transcripts, len(transcripts) > 1, was_truncated

def process_proceedings(proceedings_string):
    """Convert proceedings string into array of individual proceedings"""
    if not proceedings_string:
        return []
        
    # Split by comma and clean up each proceeding
    proceedings = [p.strip() for p in proceedings_string.split(',')]
    
    # Remove empty strings and duplicates while preserving order
    seen = set()
    return [p for p in proceedings if p and not (p in seen or seen.add(p))]

def format_house(house):
    """Validate and format House field"""
    if not house:
        return None
        
    # Normalize the input
    house = house.strip().upper()
    
    # Valid house values
    VALID_HOUSES = {
        'ASSEMBLY': 'Assembly',
        'COUNCIL': 'Council'
    }
    
    # Return properly formatted house or None if invalid
    return VALID_HOUSES.get(house)

def format_subject(subject):
    """Format subject in Title Case, handling special cases"""
    if not subject:
        return None
        
    # List of words that should remain uppercase
    uppercase_words = {'wa', 'mp', 'mlc', 'cbd', 'gst', 'abc', 'bom'}
    
    # List of words that should remain lowercase
    lowercase_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 
                      'of', 'on', 'or', 'the', 'to', 'up', 'yet'}
    
    # Split the subject into words
    words = subject.strip().lower().split()
    
    # Process each word
    for i, word in enumerate(words):
        # Always capitalize first and last word
        if i == 0 or i == len(words) - 1:
            words[i] = word.capitalize()
        # Check for special cases
        elif word in uppercase_words:
            words[i] = word.upper()
        elif word in lowercase_words:
            words[i] = word.lower()
        else:
            words[i] = word.capitalize()
    
    return ' '.join(words)

def clean_transcript_text(text):
    """Clean and format transcript text to fix common issues"""
    if not text:
        return text
        
    # Fix split words and spacing issues
    text = text.replace(' -', '-')  # Remove space before hyphen
    text = text.replace('- ', '-')  # Remove space after hyphen
    text = text.replace(' ,', ',')  # Fix space before comma
    text = text.replace(' .', '.')  # Fix space before period
    
    # Fix common split word patterns
    common_splits = {
        'ye sterday': 'yesterday',
        'a nother': 'another',
        'be cause': 'because',
        're member': 'remember',
        'to day': 'today',
        'yester day': 'yesterday',
        'every one': 'everyone',
        'some one': 'someone',
        'any one': 'anyone',
        'every thing': 'everything',
        'some thing': 'something',
        'any thing': 'anything'
    }
    
    for split_word, fixed_word in common_splits.items():
        text = text.replace(split_word, fixed_word)
    
    # Fix em-dashes and other special characters
    text = text.replace('--', '—')  # Convert double hyphen to em-dash
    text = text.replace(' - ', ' — ')  # Convert spaced hyphen to em-dash
    text = text.replace('...', '…')  # Convert triple dots to ellipsis
    text = text.replace('"', '"')  # Convert straight quotes to curly quotes
    text = text.replace('"', '"')  # Convert straight quotes to curly quotes
    text = text.replace("'", "'")  # Convert straight single quotes to curly
    text = text.replace("'", "'")  # Convert straight single quotes to curly
    
    # Fix multiple spaces
    text = ' '.join(text.split())
    
    # Fix line breaks
    text = text.replace('\r\n', '\n')  # Normalize line endings
    text = text.replace('\n\n\n', '\n\n')  # Remove excessive line breaks
    
    return text

def create_record(row, member_ids, transcript_fields, existing_records):
    """Create a new record with proper formatting"""
    global stats
    
    # Process proceedings into array
    proceedings = process_proceedings(row['Proceeding'])
    
    # Skip if Proceeding or Members are empty
    if not proceedings:
        stats['empty_proceedings'] += 1
        print(f"\nSkipping record from {row['Date']} - Empty Proceeding")
        return None
        
    if not member_ids:
        print(f"\nSkipping record from {row['Date']} - Empty or Invalid Members")
        return None  # Note: empty_members is now handled exclusively by process_members
        
    # Format House field
    formatted_house = format_house(row['House'])
    if not formatted_house:
        stats['invalid_house'] += 1  # Track invalid house values
        print(f"\nSkipping record from {row['Date']} - Invalid House value: {row['House']}")
        return None

    new_fingerprint = (
        row['Date'],
        row['Subject'],
        row['Page'],
        formatted_house,
        tuple(member_ids),
        tuple(proceedings)
    )
    
    if new_fingerprint not in existing_records:
        filesize = float(round(row['Filesize'], 2)) if pd.notnull(row['Filesize']) else None
        word_count = len(transcript_fields['Transcript'].split()) if transcript_fields['Transcript'] else 0
        
        record = {
            'fields': {
                'Date': row['Date'],
                'Page': row['Page'],
                'Subject': row['Subject'],
                'Proceeding': proceedings,
                'House': formatted_house,
                'Members': member_ids,
                'PDF': [{'url': row['PDF']}] if row['PDF'] else None,
                'PDF_URL': row['PDF'] if row['PDF'] else None,
                'Word_Count': word_count,
                'Filesize': filesize,
                **transcript_fields
            }
        }
        return record
    else:
        stats['duplicates'] += 1
        return None

# Configuration
ROWS_TO_PROCESS = 20  # Number of rows to scrape from Hansard
BATCH_SIZE = 10  # Number of records to send to Airtable at once

# URL of the Hansard webpage to scrape
target_url = "https://www.parliament.wa.gov.au/hansard/hansard.nsf/NewAdvancedSearch?openform&Query=&Fields=((%5BHan_Date%5D%3E=01/01/2021))&sWord=&sWordsSearch=&sWordAll=&sWordExact=&sWordAtLeastOne=&sMember=&sCommit=&sComms=Current&sHouse=Both%20Houses&sProc=All%20Proceedings&sPage=&sSpeechesFrom=April%202021%20-%20Current&sDateCustom=&sHansardDbs=&sYear=All%20Years&sDate=&sStartDate=&sEndDate=&sParliament=41&sBill=&sWordVar=&sFuzzy=&sResultsPerPage=100&sResultsPage=1&sSortOrd=0&sAdv=1&sRun=true&sContinue=&sWarn="

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
 ,,;;,;;;,;;;,;;;,;;;,;;;,;;;,;;,;;;,;;;,;;,;
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
                member_text = cols[3].text.strip()
                print(f"\nDebug - Member text: '{member_text}'")  # Debug output
                members.append(member_text)
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

# Initialize statistics
stats = {
    'total_scraped': 0,
    'empty_proceedings': 0,
    'empty_members': 0,
    'duplicates': 0,
    'invalid_house': 0,    # New: Track invalid house values
    'unmatched_members': set(),  # Using set to avoid duplicates
    'invalid_members': set(),     # Track members that were invalid
    'skipped_other': 0     # New: Track any other skipped records
}

# Initialize counters
skipped_records = {'empty_fields': 0, 'duplicates': 0}

# Prepare new records to be added
new_records = []
split_transcripts = []

for _, row in hansard_df.iterrows():
    stats['total_scraped'] += 1
    
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
print(f"Records skipped due to empty fields: {skipped_records['empty_fields']}")
print(f"Duplicate records skipped: {skipped_records['duplicates']}")
print(f"New records to add: {len(new_records)}")

if split_transcripts:
    print("\nSplit/Truncated Transcripts:")
    for t in split_transcripts:
        print(f"- {t['Date']}: {t['Subject']}")
        print(f"  Original length: {t['Original_Length']} characters")
        print(f"  Split into: {t['Parts']} parts")
        if t['Was_Truncated']:
            print("  Note: Transcript was truncated")

# Initialize upload results
successful = 0
failed = 0

print("\nPreparing to upload new records...")
if new_records:
    print(f"Found {len(new_records)} new records to upload")
    successful, failed = upload_records_to_airtable(new_records)
else:
    print("No new records to upload")

print("\n============ FINAL RESULTS ============")
print(f"* Total Records Scraped: {stats['total_scraped']}")
print(f"✓ Successfully uploaded: {successful} records")
print(f"✗ Failed to upload: {failed} records")

print("\nSkipped Records Breakdown:")
print(f"• Duplicates: {stats['duplicates']}")
print(f"• Empty Proceedings: {stats['empty_proceedings']}")
print(f"• Empty Members: {stats['empty_members']}")
print(f"• Invalid House Values: {stats['invalid_house']}")
print(f"• Other Skipped Records: {stats['skipped_other']}")

total_skipped = (stats['duplicates'] + stats['empty_proceedings'] + 
                stats['empty_members'] + stats['invalid_house'] + 
                stats['skipped_other'])

print(f"\nTotal Records Summary:")
print(f"• Input Records: {stats['total_scraped']}")
print(f"• Successfully Processed: {successful}")
print(f"• Total Skipped: {total_skipped}")
if stats['total_scraped'] != (successful + total_skipped):
    print(f"• Records Mismatch: {stats['total_scraped'] - (successful + total_skipped)} (investigating...)")

if stats['invalid_members']:
    print("\nInvalid Members List (roles excluded):")
    for member in sorted(stats['invalid_members']):
        print(f"  - {member}")

print(f"\nUnmatched Members (not found in database): {len(stats['unmatched_members'])}")
if stats['unmatched_members']:
    print("\nUnmatched Members List:")
    for member in sorted(stats['unmatched_members']):
        print(f"  - {member}")

print("""
                 _ _.-'`-._ _
                ;.'________'.;
     _________n.[____________].n_________
    |""_""_""_""||==||==||==||""_""_""_""]
    |.. .. .. ..||..||..||..||.. .. .. ..|
    |LI LI LI LI||LI||LI||LI||LI LI LI LI|
    |.. .. .. ..||..||..||..||.. .. .. ..|
    |LI LI LI LI||LI||LI||LI||LI LI LI LI|
 ,,;;,;;;,;;;,;;;,;;;,;;;,;;;,;;,;;;,;;;,;;,;
;;;;;;;;;  HANSARD  SCRAPE  COMPLETED  ;;;;;;
----------------------------------------------

""")
