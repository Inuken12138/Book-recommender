import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define the data types for the columns in the books.csv file
dtype_spec = {
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': object,
    'Publisher': str,
    'Description': str,
    'Categories': str,
    'Language': str,
    # Add other columns as necessary
}

# Function to fetch book info from Open Library
def get_book_info(isbn):
    url = f"https://openlibrary.org/isbn/{isbn}"
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the description part
            description_div = soup.find('div', {'class': 'book-description read-more'})
            if description_div:
                description_content = description_div.find('div', {'class': 'read-more__content'})
                description = description_content.text.strip() if description_content else 'No description available.'
            else:
                description = 'Not found'

            # Find the categories (subjects)
            subjects_div = soup.find('div', {'class': 'subjects-content'})
            if subjects_div:
                categories = []
                for link in subjects_div.find_all('a'):
                    categories.append(link.text.strip())
                categories_str = ', '.join(categories)
            else:
                categories_str = 'Not found'

            # Find the language
            language_div = soup.find('div', class_='edition-omniline')
            language = 'Not found'
            if language_div:
                language_label = language_div.find('div', class_='language')
                if language_label and language_label.text.strip() == 'Language':
                    language_span = language_label.find_next('span')
                    if language_span:
                        language_link = language_span.find('a')
                        if language_link:
                            language = language_link.text.strip()

            return {
                'ISBN': isbn,
                'Description': description,
                'Categories': categories_str,
                'Language': language
            }
        else:
            print(f"HTTP error fetching data for ISBN {isbn}: {response.status_code}")
            return {
                'ISBN': isbn,
                'Description': '',
                'Categories': '',
                'Language': ''
            }
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching data for ISBN {isbn}: {e}")
        return {
            'ISBN': isbn,
            'Description': '',
            'Categories': '',
            'Language': ''
        }

# Load the books.csv file with specified data types
# 这里改一下。对应你负责的chunk
updated_books = pd.read_csv('chuck_*.csv', dtype=dtype_spec, low_memory=False)

# Add new columns if they do not exist
for col in ['openlibrary_Description', 'openlibrary_Categories', 'openlibrary_Language']:
    if col not in updated_books.columns:
        updated_books[col] = ''

update_count = 0
# Fetch book info for each ISBN in the dataset
for index, row in updated_books.iterrows():
    if row['openlibrary_Description'] == 'Not found' or row['openlibrary_Categories'] == 'Not found' or pd.notna(row['openlibrary_Description']) or pd.notna(row['openlibrary_Categories']):
        continue
    """ if row['openlibrary_Description'] == 'Not found' or  row['openlibrary_Categories'] == 'Not found' or row['openlibrary_Language'] == 'Not found':
        continue
    elif len(row['openlibrary_Description']) > 0 or len(row['openlibrary_Categories']) > 0 or len(row['openlibrary_Language']) > 0:
        continue """
    
    isbn = row['ISBN']
    book_info = get_book_info(isbn)
    
    # Update the DataFrame with fetched information
    updated_books.at[index, 'openlibrary_Description'] = book_info['Description']
    updated_books.at[index, 'openlibrary_Categories'] = book_info['Categories']
    updated_books.at[index, 'openlibrary_Language'] = book_info['Language']
    
    update_count += 1
    # Periodically save progress
    if update_count % 100 == 0:
        update_count = 0
        updated_books.to_csv('updated_books_progress.csv', index=False)
        print(f"Progress saved at record {index + 1}")
    
    # Sleep to avoid hitting API rate limits
    #time.sleep(1)

updated_books.to_csv('updated_books_progress.csv', index=False)

print("Book summaries, categories, and language information fetched and saved to 'updated_books.csv'.")
