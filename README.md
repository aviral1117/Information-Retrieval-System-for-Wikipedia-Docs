# Information-Retrieval-System-for-Wikipedia-Docs

The following project is in the process of development as a part of the course : CS F469 - Information Retrieval (Jan'20-May'20).
Currently it contains the code related to preprocessing fundamentals required for the IR system.

## Preprocessing (Assignmnent-1)

Files: (preprocessing.py,images)

The images directory holds the frequency distributions obtained for the sample file which are later used for the infering the similarity to Zipf's law

## Vector Space Retrieval Model (Assignment-2)

Files: (index_data.py, test_queries.py)

### System Requirements: 
- Python 3.6 or higher
- NLTK
- Keras preprocessing function text_to_word_sequences

### How to run

- Please find the 'index_creation.py' file in the Code folder.

- The input corpus file should be in the same folder as the .py file. If not please change the path in the file on line no. 16.
- Run the file on the terminal with the python interpreter with the following command: '$python index_creation.py'. This will create a pickle file that stores all the required index and data structures for the IR model

- The file so created contains the following data:
    - A list of all docs containing doc id,title and content
    - Initial Index : a regular posting list from tokens to docs
    - A list of vectors of all docs in the corpus based on lnc scheme 
    - Bi-gram character level index on the corpus (eg: bo---> board,bord,boardroom...)
    - Modified posting list based on the term frequency values for champion list creation (champion list criteria size=5)

- After executing the above file use the following command to test the queries:
    $python test_queries.py "sample_query" "index_data_path_created_above"


- Run the following command inside the python environment 'nltk.download()' to download the required nltk packages in case of error.

- Please follow the print statement outputs on running the code.
