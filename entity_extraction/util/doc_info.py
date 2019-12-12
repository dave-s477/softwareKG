import re
import xml.etree.ElementTree as ET

def get_doc_info(fake_doi):
    tree = ET.parse('data/xml_data/{}.xml'.format(fake_doi))
    root = tree.getroot()
    front = root[0]
    journal_meta = front[0]
    journal_title_group_node = journal_meta.find('journal-title-group')
    journal_title = ''
    if journal_title_group_node != None:
        journal_title_node = journal_title_group_node.find('journal-title')
        if journal_title_node != None:
            journal_title = journal_title_node.text
    publisher = journal_meta.find('publisher').find('publisher-name').text
    article_meta = front[1]
    article_id = article_meta.findall('article-id')
    for x in article_id:
        if x.attrib['pub-id-type'] == 'doi':
            doi = x.text
    subj_groups = article_meta.find('article-categories').findall('subj-group')
    headings = set()
    subjects = set()
    for x in subj_groups:
        if 'subj-group-type' in x.attrib:
            if x.attrib['subj-group-type'] == 'heading':
                h = x.findall('subject')
                for value in h:
                    headings.add(value.text)
            elif x.attrib['subj-group-type'] == 'Discipline':
                s = x.findall('subject')
                for value in s:
                    subjects.add(value.text)
            elif x.attrib['subj-group-type'] == 'Discipline-v2':
                s = x.findall('subject')
                for value in s:
                    subjects.add(value.text)
            elif x.attrib['subj-group-type'] == 'Discipline-v3':
                s = x.findall('subject')
                for value in s:
                    subjects.add(value.text)
    title = re.sub(r'<[^<]+>', "", str(ET.tostring(article_meta.find('title-group').find('article-title')), 'utf-8')) #article_meta.find('title-group').find('article-title').text
    authors = []
    affiliation_dict = {}
    if article_meta.find('contrib-group') != None:
        contribs = article_meta.find('contrib-group').findall('contrib')
        affiliations = article_meta.findall('aff')
        for affiliation in affiliations:
            if affiliation.attrib['id'].startswith("aff"):
                affiliation_dict[affiliation.attrib['id']] = affiliation.find('addr-line').text
        for idx, x in enumerate(contribs):
            name = x.find('name')
            authors.append({})
            authors[idx]['@id'] = 'http://data.gesis.org/softwarekg/' + doi + '/authors/' + str(idx)
            authors[idx]['@type'] = "http://schema.org/Person"
            if name != None:
                surname = ''
                surname_node = name.find('surname')
                if surname_node != None:
                    surname = surname_node.text
                #authors[idx]['surname'] = surname
                authors[idx]['http://schema.org/familyName'] = surname
                given_name = ''
                given_name_node = name.find('given-names')
                if given_name_node != None:
                    given_name = given_name_node.text
                #authors[idx]['given-name'] = given_name
                authors[idx]['http://schema.org/givenName'] = given_name
                orcid = ''
                contrib_ids = x.findall('contrib-id')
                for contrib_id in contrib_ids:
                    if contrib_id.attrib['contrib-id-type'] == 'orcid':
                        orcid = contrib_id.text
                if orcid:
                    if "orcid.org/" in orcid:
                        orcid = orcid.split("orcid.org/")[-1]
                    authors[idx]['http://data.gesis.org/softwarekg/orcid'] = orcid
            authors[idx]['http://schema.org/affiliation'] = []
            xrefs = x.findall('xref')
            if xrefs != None:
                for xref in xrefs:
                    if xref.attrib['ref-type'] == 'aff':
                        authors[idx]['http://schema.org/affiliation'].append(affiliation_dict[xref.attrib['rid']])
        
    date = article_meta.findall('pub-date')
    for x in date:
        if x.attrib['pub-type'] == 'epub':
            day = x.find('day').text
            month = x.find('month').text
            year = x.find('year').text
    publication_day = '{}.{}.{}'.format(day, month, year)
    counts = article_meta.find('counts')
    pages = ''
    if counts != None:
        page_counts = counts.find('page-count')
        pages = page_counts.attrib['count']
    return doi, journal_title, publisher, list(headings), list(subjects), title, authors, publication_day, pages

def get_software_info(candidates, doi):
    info_list = []
    for idx, cand in enumerate(candidates):
        info_list.append({})
        info_list[idx] = {
            '@id': 'http://data.gesis.org/softwarekg/' + doi + '/software/' + str(idx),
            '@type': 'http://schema.org/SoftwareApplication',
            'http://schema.org/name': cand,
            'http://data.gesis.org/softwarekg/open_source': 'unk',
            'http://schema.org/license': 'unk'
        }
    return info_list

def get_doc_dict(fake_doi, candidates):
    doi, journal_title, publisher, headings, subjects, title, authors, publication_day, pages = get_doc_info(fake_doi)
    candidate_list = get_software_info(candidates, doi)
    year = publication_day.split('.')[-1]
    graph_entry = {
        "@id": "http://data.gesis.org/softwarekg/" + doi,
        "@type": "http://schema.org/ScholarlyArticle",
        "http://schema.org/identifier": doi,
        "http://data.gesis.org/softwarekg/journalTitle": journal_title,
        "http://schema.org/publisher": publisher,
        "http://data.gesis.org/softwarekg/headings": headings,
        "http://schema.org/keywords": subjects,
        "http://schema.org/name": title,
        "http://schema.org/author": authors, 
        "http://schema.org/datePublished": publication_day,
        "http://schema.org/numberOfPages": pages,
        "http://data.gesis.org/softwarekg/software": candidate_list,
        "http://purl.org/dc/elements/1.1/date": year
    }
    # graph_entry = {
    #     'doi': doi,
    #     'journal_title': journal_title,
    #     'publisher': publisher,
    #     'headings': headings,
    #     'subjects': subjects,
    #     'title': title,
    #     'authors': authors,
    #     'publication_day': publication_day,
    #     'pages': pages, 
    #     'software': candidate_dict
    # }
    return graph_entry