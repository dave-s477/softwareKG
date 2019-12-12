import re
import string
from snorkel.lf_helpers import (
    contains_token, get_between_tokens, get_doc_candidate_spans,
    get_left_tokens, get_matches, get_right_tokens, 
    get_sent_candidate_spans, get_tagged_text, get_text_between, 
    get_text_splits, is_inverted
)

def dynamic_growing_right_context(c, max_size=4):
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    top1_head_words = ['statistical', 
                       'method', 
                       'procedure', 
                       'kit', 
                       'version',
                       'Version',
                       'v',
                       'V',
                       'v.',
                       'V.',
                       'ver.']
    right_context = [x for x in get_right_tokens(c, window=max_size+1, attrib="lemmas")]
    for i,token in enumerate(right_context):
        if i == max_size-1:
            return right_context[max_size:max_size+1]
        if (token in string.punctuation or 
            token in top1_head_words or
            version_number.match(token)):
            pass
        else:
            return right_context[i:i+1]
        
def LF_pan_top_1(c, stopwords):
    '''use <> software'''
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    top1_head_words = ['statistical', 
                       'method', 
                       'procedure', 
                       'kit', 
                       'version',
                       'Version',
                       'v',
                       'V',
                       'v.',
                       'V.',
                       'ver.']
    left_context = ['use']
    right_context = ['software']
    tokens = [x.lower() for x in c[0].get_attrib_tokens()]
    
    if tokens[0] in stopwords or 'computer' in tokens or 'custom' in tokens or 'and' in tokens or tokens[-1] in ['statistical']:
        return -1 
    for tok in tokens:
        if tok in string.punctuation or tok in top1_head_words or version_number.match(tok):
            return -1
    left_win_1 = [x for x in get_left_tokens(c, window=1, attrib="lemmas")]
    left_win_2 = [x for x in get_left_tokens(c, window=2, attrib="lemmas")]
    if len(left_win_2) > 0 and left_win_2[-1] in stopwords:
        left_features = left_win_2[:-1]
    else:
        left_features = left_win_1
    right_features = dynamic_growing_right_context(c)
    if not right_features or len(left_context) != len(left_features) or len(right_context) != len(right_features):
        return 0
    for cont,feat in zip(left_context, left_features):
        if cont != feat:
            return 0
    for cont,feat in zip(right_context, right_features):
        if cont != feat:
            return 0
    return 1

def LF_pan_top_2(c,stopwords):
    '''perform use <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['perform', 'use']
    left_features = [x for x in get_left_tokens(c, window=2, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_3(c, stopwords):
    '''be perform use <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['be', 'perform', 'use']
    left_features = [x for x in get_left_tokens(c, window=3, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_4(c, stopwords):
    '''analysis be perform use <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['analysis', 'be', 'perform', 'use']
    left_features = [x for x in get_left_tokens(c, window=4, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_5(c, stopwords):
    '''analyze use <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'unpaired',
                           'one-way',
                           'two-way',
                           'anova',
                           't-test',
                           'chi-square',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['analyze', 'use']
    left_features = [x for x in get_left_tokens(c, window=2, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_6(c, stopwords):
    '''analysis be perform with <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['analysis', 'be', 'perform', 'with']
    left_features = [x for x in get_left_tokens(c, window=4, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def dynamic_growing_context_top7(c, max_size=4, context=2, debug=False):
    top7_head_words = ['version',
                       'Version',
                       'v',
                       'V',
                       'v.',
                       'V.',
                       'ver.']
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    right_context = [x for x in get_right_tokens(c, window=max_size+context, attrib="lemmas")]
    if debug:
        print("right context")
        print(right_context)
    for i,token in enumerate(right_context):
        if i == max_size-1:
            if debug:
                print("Return")
                print(right_context[max_size:max_size+context])
            return right_context[max_size:max_size+context]
        if (token in string.punctuation or 
            token in top7_head_words or
            version_number.match(token)):
            if debug:
                print("passed up token "+ token)
            pass
        else:
            if debug:
                print("Return")
                print(right_context[i:i+context])
            return right_context[i:i+context]

def LF_pan_top_7(c, stopwords):
    '''<> statistical software'''
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    for tok in tokens:
        if tok == 'use' or tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    debug = False
    right_context = ['statistical', 'software']
    right_features = dynamic_growing_context_top7(c, debug=debug)
    if not right_features or len(right_context) != len(right_features):
        return 0
    for c,f in zip(right_context, right_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_8(c, stopwords):
    '''<> software be use'''
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    for tok in tokens:
        if tok == 'use' or tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    right_context = ['software', 'be', 'use']
    right_features = dynamic_growing_context_top7(c, context=3, debug=False)
    if not right_features or len(right_context) != len(right_features):
        return 0
    for c,f in zip(right_context, right_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_9(c, stopwords):
    '''quantify use <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['quantify', 'use']
    left_features = [x for x in get_left_tokens(c, window=2, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def LF_pan_top_10(c, stopwords):
    '''be caclulate use <>'''
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    for tok in tokens:
        if tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    left_context = ['be', 'calculate', 'use']
    left_features = [x for x in get_left_tokens(c, window=3, attrib="lemmas")]
    if len(left_context) != len(left_features):
        return 0
    for c,f in zip(left_context, left_features):
        if c != f:
            return 0
    return 1

def get_normalized_next_right_word(c, max_size=6):
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    positive_head_nouns = [
        'software',
        'package',
        'program', 
        'tool', 
        'toolbox',
        'web-service',
        'spreadsheet'
    ]
    top1_head_words = ['statistical', 
                       'method', 
                       'procedure', 
                       'kit', 
                       'version',
                       'Version',
                       'v',
                       'V',
                       'v.',
                       'V.',
                       'ver.']
    right_context = [x for x in get_right_tokens(c, window=max_size, attrib="lemmas")]
    for i,token in enumerate(right_context):
        if token in positive_head_nouns:
            return 1
        elif (token in string.punctuation or 
            token in top1_head_words or
            version_number.match(token)):
            pass
        else:
            return 0
        
def LF_software_head_nouns(c, stopwords):
    negative_head_words = ['statistical', 
                           'software', 
                           'method', 
                           'procedure', 
                           'kit', 
                           'program', 
                           'tool', 
                           'toolbox',
                           'version',
                           'Version',
                           'v',
                           'V',
                           'v.',
                           'V.',
                           'ver.']
    tokens = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    for tok in tokens:
        if tok in ['software', 'program', 'tool', 'computer', 'custom'] or tok in stopwords or tok in string.punctuation or tok in negative_head_words or re.match(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)', tok):
            return -1
    poses = [x for x in c[0].get_attrib_tokens(a="pos_tags")]
    for pos in poses:
        if pos not in ['NN', 'NNS', 'NNP', 'NNPS']:
            return -1
    left_context = [x for x in get_left_tokens(c, window=1, attrib="pos_tags")]
    left_words = [x for x in get_left_tokens(c, window=1, attrib="words")]
    if left_context and left_context[0] in ['nn', 'nns', 'nnp', 'nnps']:
        return 0 # -1
    res = get_normalized_next_right_word(c)
    if res:
        return res
    else:
        return 0
    
def LF_version_number(c):
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    simple_float = re.compile(r'^0\.\d{1,3}$')
    common_measures = ['nm', 'µm', 'mm', 'cm', 'dm', 'm', 'km', 'mg', 'g', 'kg', 'ml', 'l', 's', 'h', 'y']
    restrictive_version_number = re.compile(r'^(v|V|v.|V.)?(\d{1,3}\.)?(\d{1,3}\.)(\d{1,3})$')
    lemmas = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    pos_tags = [x for x in c[0].get_attrib_tokens(a="pos_tags")]
    for lem in lemmas:
        if (len(lem) <= 1 and not lem.isalpha()) or lem in ['v', 'V', 'v.', 'V.', 'ver.', 'Ver.', 'version', 'Version'] or lem in ['software', 'package', 'program'] or lem == 'ph':
            return -1
    for pos in pos_tags:
        if pos not in ['NN', 'NNS', 'NNP', 'NNPS']:
            return -1
    right_context = [x for x in get_right_tokens(c, window=4, attrib="lemmas")]
    to_examine = 0
    if not right_context:
        return 0
    if right_context[0] in ['(']:#,'[','{']: #TODO: also exclude software, package, software package, etc.
        to_examine = 1
    if len(right_context) > 1 and right_context[to_examine] in ['v', 'V', 'v.', 'V.', 'ver.', 'Ver.', 'version', 'Version']:
        if len(right_context) > 2:
            potential_version_number = right_context[to_examine+1]
            if version_number.match(potential_version_number):
                return 1
        return 0
    if len(right_context) > 1 and not simple_float.match(right_context[to_examine]) and restrictive_version_number.match(right_context[to_examine]):
        if len(right_context) > 2:
            next_right_context = right_context[to_examine+1]
            if next_right_context in ['%'] or next_right_context in common_measures:
                return -1
        return 1
    return 0

def LF_url(c):
    # Overkill??
    url_regex = re.compile(r"^((http(s)?:\/\/www\.)|(http(s)?:\/\/)|(www\.))[a-z\.-]+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$")
    version_indicators = ['v', 'V', 'v.', 'V.', 'ver.', 'Ver.', 'version', 'Version']
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    positive_head_nouns = [
        'software',
        'package',
        'program', 
        'tool', 
        'toolbox',
        'web-service',
        'spreadsheet'
    ]
    lemmas = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    pos_tags = [x for x in c[0].get_attrib_tokens(a="pos_tags")]
    for lem in lemmas:
        if (len(lem) <= 1 and not lem.isalpha()) or version_number.match(lem) or lem in version_indicators or lem in positive_head_nouns or lem in ['computer']:
            return -1
    for pos in pos_tags:
        if pos not in ['NN', 'NNS', 'NNP', 'NNPS']:
            return -1
    left_context = [x for x in get_left_tokens(c, window=1, attrib="pos_tags")]
    left_words = [x for x in get_left_tokens(c, window=1, attrib="words")]
    if left_context and left_context[0] in ['nn', 'nns', 'nnp', 'nnps']:
        return 0 # -1
    # In principle we just want to look at the entire right context and decide for our selfs how big it should be
    right_context = [x for x in get_right_tokens(c, window=15, attrib="lemmas")]
    right_context_size = len(right_context)
    first_token_to_consider = 0
    while (first_token_to_consider < right_context_size and 
           (right_context[first_token_to_consider] in version_indicators or 
            right_context[first_token_to_consider] in positive_head_nouns or right_context[first_token_to_consider] in ['computer'] or 
            version_number.match(right_context[first_token_to_consider]))):
        first_token_to_consider += 1 
        
    if first_token_to_consider == right_context_size or right_context[first_token_to_consider] != '(':
        return 0
    else:
        remaining_right_context = right_context[first_token_to_consider+1:]
        while remaining_right_context:
            tok = remaining_right_context.pop(0)
            if tok == ')':
                return 0
            if url_regex.match(tok):
                return 1
        return 0 

def LF_developer(c):
    version_number = re.compile(r'(v|V|v.|V.)?(\d+\.)?(\d+\.)?(\d+)')
    developer_version_addition = ['v.', 'ver.', 'version']
    version_indicators = ['v', 'V', 'v.', 'V.', 'ver.', 'Ver.', 'version', 'Version']
    positive_head_nouns = [
        'software',
        'package',
        'program', 
        'tool', 
        'toolbox',
        'web-service',
        'spreadsheet'
    ]
    lemmas = [x.lower() for x in c[0].get_attrib_tokens(a="lemmas")]
    pos_tags = [x for x in c[0].get_attrib_tokens(a="pos_tags")]
    for lem in lemmas:
        if (len(lem) <= 1 and not lem.isalpha()) or version_number.match(lem) or lem in version_indicators or lem in positive_head_nouns or lem in ['computer']:
            return -1
    for pos in pos_tags:
        if pos not in ['NN', 'NNS', 'NNP', 'NNPS']:
            return -1
    left_context = [x for x in get_left_tokens(c, window=1, attrib="pos_tags")]
    left_words = [x for x in get_left_tokens(c, window=1, attrib="words")]
    if left_context and left_context[0] in ['nn', 'nns', 'nnp', 'nnps']:
        return 0 # -1
    # In principle we just want to look at the entire right context and decide for our selfs how big it should be
    right_context = [x for x in get_right_tokens(c, window=20, attrib="lemmas")]
    right_context_size = len(right_context)
    first_token_to_consider = 0
    while (first_token_to_consider < right_context_size and 
           (right_context[first_token_to_consider] in version_indicators or 
            right_context[first_token_to_consider] in positive_head_nouns or right_context[first_token_to_consider] in ['computer'] or 
            version_number.match(right_context[first_token_to_consider]))):
        first_token_to_consider += 1 
    # Behave different for here: We want to examine the context, therefore we first extract the entire context 
    # by looking for the closing bracket first. 
    
    if first_token_to_consider == right_context_size or right_context[first_token_to_consider] != '(':
        return 0
    else:
        remaining_tokens = right_context[first_token_to_consider+1:]
        #remaining_words = right_words[first_token_to_consider+1:]
        last_token_to_consider = -1
        for i,tok in enumerate(remaining_tokens):
            if tok == ')':
                last_token_to_consider = i 
                break
        if last_token_to_consider < 0:
            return 0
        else: 
            remaining_tokens = remaining_tokens[:last_token_to_consider]
            #remaining_words = remaining_words[:last_token_to_consider]
            # Here we perform the actual test
            for tok in remaining_tokens:
                if tok in developer_version_addition or tok in ['inc', 'ltd', 'corp', 'apply']:
                    return 1
                if tok in ['such', 'i.e', 'e.g']:
                    return -1
            #for tok in remaining_words:
            #    if tok in us_states:
            #        return 1
            token_split = [[]]
            for i in remaining_tokens:
                if i in [',', ';']:
                    token_split.append([])
                else:
                    token_split[-1].append(i)
            
            return 0
            
def LF_distant_supervision(c, software_dict, software_dict_lower, english_dict, english_dict_lower, acronym_dict, gen_seqs):
    cand = c[0].get_span()
    tokens = [x.lower() for x in c[0].get_attrib_tokens()]
    if len(tokens) == 1 and len(tokens[0]) != len(cand):
        return 0
    omissions = ['California', 'NaCl', 'control groups', 'FID', 'ELISA', 'GPs', 'PubMed', 'Gaussian', 'synaptic', 'vivo', 'ionic']
    if cand in omissions:
        return -1
    if len(cand) == 2 or cand.isdigit() or all(char in string.punctuation for char in cand):
        return -1
    cand_lower = cand.lower()
    cand_in_known_software = cand in software_dict
    cand_in_english_dic = cand in english_dict # english_dict
    cand_lower_match_known_software = cand_lower in software_dict_lower
    cand_lower_match_english_dic = cand_lower in english_dict_lower # english_dict_lower
    cand_is_acronym = cand in acronym_dict
    cand_is_gen_seq = cand in gen_seqs
    
    left_tokens = [x for x in get_left_tokens(c, window=1)]
    right_tokens = [x for x in get_right_tokens(c, window=1)]

    if ('institutional' in left_tokens or 
        'institution' in left_tokens or 
        'ethics' in left_tokens or 
        'ethic' in left_tokens or 
        (len(left_tokens) > 0 and len(right_tokens) > 0 and left_tokens[-1] in ['(', '[', '{'] and right_tokens[0] in [')', ']', '}'])):
        return -1
    
    if cand_is_gen_seq:
        return -1
    if cand_in_english_dic:
        if cand_in_known_software:
            return 0
        else: 
            return -1
    else:
        if cand_in_known_software:
            return 1
        elif cand_lower_match_known_software:
            return 1
        elif cand_lower_match_english_dic:
            return 0 # -1 
        else:
            return 0