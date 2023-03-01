import re
from collections import Counter
from langdetect import detect

def numbered_reference_removal(text):
    # refs are typically in the form [n] in the text.
    matches = re.findall(r'\[[0-9]+?\]', text)
    counter = Counter(matches)
    for match in matches:
        if counter[match] != 2:
            # print(f"Didn't find a reference twice in the text, but {counter[match]} times. Cannot remove.")
            return text
    
    assert len(matches)%2 == 0
    if len(matches) == 0:
        return text
    N = len(matches)//2
    is_ordered = True
    n = 0
    for s in matches:
        is_ordered = (s == f'[{n%N+1}]')
        if not is_ordered : 
            # print(f"Not ordered, {str(matches)}")
            return text
        n += 1
    
    # Remove all references after [1]
    res = re.findall(
        r'^.*?\[1\].+?\[1\]',
        text
    )
    return res[0]

def reference_removal_en(text):
    res = re.sub(r'(?i)see also.*', '', text)
    res = re.sub(r'References.*', '', res)
    res = re.sub("\[[0-9]+\]", "", res)
    return res

# Before dealing with N/A because otherwise, this edit would just undo the other.
# We can use this to remove the "TRANSCRIPT" and "SPEECH" tags.
# We also notice that most of the time, the content is preceded by a date.
# It would be useful to make a date parser to remove the first chunk.

def first_date_extractor(text):
    if len(text) > 0:
        res = re.sub('^(.*?)[1-9][0-9]* (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
                     + 'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?) (19|20)[0-9][0-9]',
                     '', text)
        return res

def remove_title(x):
    if not x["text"] is None and isinstance(x["text"], str):
        res = re.sub(x["title"], '', x["text"]).strip()
        return res
    else:
        return x["title"]

def website_remover(text):
    # Remove websites
    regex = "((http|https)://)(www.)?" \
        + "[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]" \
        + "{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)"
    res = re.sub(regex, "", text).strip()
    return res

def tag_removal(text):
    # Remove tags
    res = re.sub('SPEECH', '', text)
    res = re.sub('TRANSCRIPT', '', res)
    res = re.sub("Introduction", "", res)
    res = re.sub("Summary", "", res)
    return res

def ecb_pipeline_en(x):
    res = remove_title(x)
    if res is None:
        return res
    res = numbered_reference_removal(res)
    res = reference_removal_en(res)
    res = tag_removal(res).strip()
    res = first_date_extractor(res).strip()

    return res

def fast_detect(x, bound=500):
    return detect(x[:min(len(x), bound)])