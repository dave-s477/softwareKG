# Expects a query and uses it to access WikiData

import sys, getopt
import urllib.request as url
import re
import requests
import json
import pandas as pd 

from urllib.parse import quote_plus
from io import StringIO

def parse_query(query_file, lang):
    """
    Create a valid WikiData query URL from a text file.
    
    Arguments:
        query_file {[type]} -- [query file in text form]
        lang {[type]} -- [language in which to access WikiData]
    
    Returns:
        [str] -- [query url]
    """
    with open(query_file, 'r') as query_handle:
        query_content = query_handle.read()
        _, plain_query, _ = query_content.split("::", 2)
        plain_query = re.sub('"en"', '"' + lang + '"', plain_query)
        encoded_query = quote_plus(plain_query, safe=r"()\{\}")
        encoded_query = re.sub(r"\+", "%20", encoded_query)
        base = "https://query.wikidata.org/bigdata/namespace/wdq/sparql?query="
        complete_url = base + encoded_query
        print("Querying the results for:")
        print(complete_url)
        return complete_url

def execute_query(query_url):
    """
    Execute WikiData query, accepted is CSV output.
    
    Arguments:
        query_url {[type]} -- [WikiData query URL]
    
    Returns:
        [csv] -- [csv result from WikiData]
    """
    headers = {
        'Accept': 'text/csv'
    }
    print("Loading results")
    response = requests.get(query_url, headers=headers)
    if response.status_code != 200:
        print("Received HTTP-Statuscode other than 200 - Aborting" + str(response.status_code))
        if response.status_code == 400:
            print("Error is probably in the query syntax.")
        sys.exit(1)
    else:
        return response.text

def retrieve_results(query_file, lang):
    query = parse_query(query_file, lang)
    return execute_query(query)

def main(argv):
    query_source = ''
    try:
        opts, _ = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('Usage: perform_wiki_data_queries.py -i <wikidata-query>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: perform_wiki_data_queries.py -i <wikidata-query>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            query_source = arg
    if not query_source:
        print('Valid input file expected. Usage: perform_wiki_data_queries.py -i <wikidata-query>')
        sys.exit(2)

    languages = ["en", "de", "fr", "es"]
    # Bad lines are skipped: about 3 out of 24000 entries are corrupted (most likely contain unescaped commas, leading to confusion when parsing the csv file..), we are just going to ignore them here
    csv = pd.read_csv(StringIO(retrieve_results(query_source, languages[0])), error_bad_lines=False)
    print("Retrieved " + str(csv['item'].count()) + " rows in " + languages[0])
    for lang in languages[1:]:
        tmp_csv = pd.read_csv(StringIO(retrieve_results(query_source, lang)), error_bad_lines=False)
        csv = csv.append(tmp_csv)
        print("Retrieved " + str(tmp_csv['item'].count()) + " rows in " + lang)
        print("Total " + str(csv['item'].count()) + " rows.\n")
    out_file = "wikidata_query_result"
    print("Writing output to " + out_file)
    csv.to_csv(out_file + '.csv', index = False, encoding='utf-8')

if __name__ == "__main__":
   main(sys.argv[1:])